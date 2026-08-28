"""
The whole scene, in ten draw calls.

::

    1  particle solver         (fragment pass over a 128x128 state pair)
    2  camera frame
    3  rune circles            (instanced, one per hand)
    4  lightning bolts
    5  particles               (instanced, one draw for 16,384)
    6  the target word
    7  bloom downsample        (quarter res)
    8  bloom blur X
    9  bloom blur Y
    10 composite to the screen

Nothing in frame() allocates. Every buffer, texture and array below is built
during init.

The 2D chrome the browser build kept in the DOM is drawn by ui/shell.py after
this, as scissored bands from one texture plus the charge bar — see there for why.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import moderngl
import numpy as np

from ..config import ELEMENT_COLOR, ELEMENT_HOT, FX, Element
from .bolts import BoltField
from .gl import FULLSCREEN_VS, FullscreenTriangle, Program, Window, create_framebuffer, create_texture
from .shaders.particles import DRAW_FS, DRAW_VS, SIM_FS
from .shaders.post import BLIT_FS, BLOOM_BLUR_FS, BLOOM_DOWN_FS, COMPOSITE_FS
from .shaders.scene import (
    BOLT_FS,
    BOLT_VS,
    RUNE_FS,
    RUNE_VS,
    VIDEO_FS,
    VIDEO_VS,
    WORD_FS,
    WORD_VS,
)
from .video_texture import VideoTexture
from .view import View
from .word_texture import WordTexture

MODE_CHARGE = 0
MODE_BURST = 1

CAM_Z = 2.2
OVERSCAN = 1.06
FAIL_COLOR: Tuple[float, float, float] = (1.0, 0.24, 0.5)

MASK32 = 0xFFFFFFFF


@dataclass
class HandFx:
    """Everything the app tells the renderer about one hand, mutated in place."""

    active: bool = False
    #: Palm centre in world space.
    x: float = 0.0
    y: float = 0.0
    rune_size: float = 0.28
    #: 0..1 hold progress. Drives rune scale, brightness and spin.
    charge: float = 0.0
    #: Rune opacity, faded independently so it can linger through a cast.
    alpha: float = 0.0
    #: 0..1 shatter-and-fizzle amount for a failed cast.
    fail: float = 0.0
    mode: int = MODE_CHARGE
    spawn_id: float = 0.0
    power: float = 0.5


class Renderer:
    def __init__(self, window: Window) -> None:
        self.window = window
        self.ctx = window.ctx
        self.caps = window.caps
        self.view = View()
        self.video = VideoTexture(self.ctx)
        self.word_tex = WordTexture(self.ctx)
        self.fx: List[HandFx] = [HandFx(), HandFx()]

        #: Which element is loaded. Both hands always cast the same one.
        self.element: Element = Element.FIRE
        #: Combo multiplier, 0-based. Larger effects, hotter colours.
        self.combo = 0
        #: Global effect intensity, from the settings drawer.
        self.intensity = 1.0
        #: Set while a cast is failing, which recolours everything magenta.
        self.failing = False
        #: Display mirroring. The landmark pipeline is *always* mirrored — that
        #: invariant is what the unit tests protect — so turning this off flips
        #: the whole composition instead: the video's u axis and the
        #: landmark-to-world mapping change sign together and stay in agreement.
        self.mirror = True

        self.word_alpha = 0.0
        self.word_dissolve = 0.0
        self.word_scale = 1.0

        self.draw_calls = 0
        self.particles_drawn = 0
        self.low_power = False

        self.tex_size = FX.particle_tex
        self.max_particles = self.tex_size * self.tex_size

        self.tri = FullscreenTriangle(self.ctx)
        self.bolts = BoltField(self.ctx)

        # Programs.
        self.p_sim = Program(self.ctx, FULLSCREEN_VS, SIM_FS, "sim")
        self.p_draw = Program(self.ctx, DRAW_VS, DRAW_FS, "particles")
        self.p_video = Program(self.ctx, VIDEO_VS, VIDEO_FS, "video")
        self.p_rune = Program(self.ctx, RUNE_VS, RUNE_FS, "rune")
        self.p_bolt = Program(self.ctx, BOLT_VS, BOLT_FS, "bolt")
        self.p_word = Program(self.ctx, WORD_VS, WORD_FS, "word")
        self.p_bloom_down = Program(self.ctx, FULLSCREEN_VS, BLOOM_DOWN_FS, "bloom-down")
        self.p_bloom_blur = Program(self.ctx, FULLSCREEN_VS, BLOOM_BLUR_FS, "bloom-blur")
        self.p_composite = Program(self.ctx, FULLSCREEN_VS, COMPOSITE_FS, "composite")
        self.p_blit = Program(self.ctx, FULLSCREEN_VS, BLIT_FS, "blit")

        for program in (
            self.p_sim,
            self.p_bloom_down,
            self.p_bloom_blur,
            self.p_composite,
            self.p_blit,
        ):
            self.tri.bind(program)
        self.bolts.bind(self.p_bolt)

        # Geometry.
        self._corner_vbo = self.ctx.buffer(
            np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4").tobytes()
        )
        self._vao_video = self.p_video.vao([(self._corner_vbo, "2f", "aCorner")], self.ctx)
        self._vao_rune = self.p_rune.vao([(self._corner_vbo, "2f", "aCorner")], self.ctx)
        self._vao_word = self.p_word.vao([(self._corner_vbo, "2f", "aCorner")], self.ctx)
        self._vao_particles = self._build_particle_vao()
        self.spawn_tex = self._build_spawn_texture()

        # Particle state, ping-ponged.
        self.pos_tex: List[moderngl.Texture] = []
        self.vel_tex: List[moderngl.Texture] = []
        self.sim_fbo: List[moderngl.Framebuffer] = []
        self._build_particle_state()
        self._ping_pong = 0

        # Render targets.
        self.scene_tex: Optional[moderngl.Texture] = None
        self.scene_fbo: Optional[moderngl.Framebuffer] = None
        self.bloom_tex: List[moderngl.Texture] = []
        self.bloom_fbo: List[moderngl.Framebuffer] = []
        self._present_tex: Optional[moderngl.Texture] = None
        self._present_fbo: Optional[moderngl.Framebuffer] = None
        self._readback_buffers: List[moderngl.Buffer] = []
        self._readback_index = 0
        self._readback_ready = False
        self.buffer_w = 1
        self.buffer_h = 1
        self.bloom_w = 1
        self.bloom_h = 1
        self._screen_rect = (0, 0, 1, 1)
        #: While a clip is recording the composite goes to an offscreen target at
        #: the export resolution and is blitted to the window letterboxed.
        self.capture = False

        # Preallocated uniform payloads. Uploaded every frame, never reallocated.
        self._u_palm = np.zeros(6, dtype=np.float32)
        self._u_spawn_id = np.zeros(2, dtype=np.float32)
        self._u_mode = np.zeros(2, dtype=np.int32)
        self._u_power = np.zeros(2, dtype=np.float32)
        self._u_charge = np.zeros(2, dtype=np.float32)
        self._u_rune_pos = np.zeros(4, dtype=np.float32)
        self._u_rune_size = np.zeros(2, dtype=np.float32)
        self._u_rune_alpha = np.zeros(2, dtype=np.float32)
        self._u_rune_fail = np.zeros(2, dtype=np.float32)
        self._color = np.zeros(3, dtype=np.float32)
        self._hot = np.zeros(3, dtype=np.float32)
        self._packed = 0

        self.aspect = 16.0 / 9.0
        self._cover_x = 1.0
        self._cover_y = 1.0
        self._time = 0.0

        # Bolt cadence. Lightning is regenerated a handful of times per cast,
        # which is what makes it crackle; it is never rebuilt per frame.
        self._bolt_timer = 0.0
        self._bolt_life = 0.0
        self._bolt_x = 0.0
        self._bolt_y = 0.0
        self._bolt_reach = 0.5
        self._bolt_power = 1.0

        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.CULL_FACE)

    # ---------------------------------------------------------------- setup ---

    def _build_particle_vao(self) -> moderngl.VertexArray:
        """
        The static per-instance attribute: each particle's own texel in the state
        textures. Built once; the fire event never touches a buffer.
        """
        uv = np.empty(self.max_particles * 2, dtype=np.float32)
        inv = 1.0 / self.tex_size
        idx = np.arange(self.max_particles, dtype=np.float32)
        uv[0::2] = (idx % self.tex_size + 0.5) * inv
        uv[1::2] = (idx // self.tex_size + 0.5) * inv
        self._uv_vbo = self.ctx.buffer(uv.tobytes())
        return self.p_draw.vao(
            [
                (self._corner_vbo, "2f", "aCorner"),
                (self._uv_vbo, "2f/i", "aParticleUv"),
            ],
            self.ctx,
        )

    def _build_spawn_texture(self) -> moderngl.Texture:
        """
        Burst directions, precomputed into a static float texture at init.

        This is what makes a cast free: the solver reads a particle's direction
        and seed from here, so firing only has to change a uniform. Nothing is
        generated, allocated or uploaded at cast time.
        """
        data = np.zeros(self.max_particles * 4, dtype=np.float32)
        rng = 0x12345678

        def rand() -> float:
            nonlocal rng
            rng ^= (rng << 13) & MASK32
            rng ^= rng >> 17
            rng ^= (rng << 5) & MASK32
            rng &= MASK32
            return (rng & 0xFFFFFF) / 0x1000000

        for i in range(self.max_particles):
            # Uniform on the sphere, then biased hard toward the camera so the
            # burst comes at the viewer instead of spreading sideways.
            u = rand() * 2.0 - 1.0
            th = rand() * math.pi * 2.0
            s = math.sqrt(max(0.0, 1.0 - u * u))
            dx = s * math.cos(th)
            dy = s * math.sin(th)
            # Biased toward the camera so the burst comes at the viewer, but not
            # so hard that every particle piles into the middle of the lens.
            dz = abs(u) * 0.55 + 0.26
            length = math.sqrt(dx * dx + dy * dy + dz * dz) or 1.0
            data[i * 4] = dx / length
            data[i * 4 + 1] = dy / length
            data[i * 4 + 2] = dz / length
            data[i * 4 + 3] = rand()

        return create_texture(
            self.ctx, self.tex_size, self.tex_size, 4, "f4", data.tobytes(), moderngl.NEAREST
        )

    def _build_particle_state(self) -> None:
        # RGBA32F. Half float positions quantise to about 0.001 of a world unit,
        # which is visible as stepping on a slow-moving ember.
        dtype = "f4" if self.caps.color_float else "f2"
        for _ in range(2):
            pos = create_texture(self.ctx, self.tex_size, self.tex_size, 4, dtype)
            vel = create_texture(self.ctx, self.tex_size, self.tex_size, 4, dtype)
            self.pos_tex.append(pos)
            self.vel_tex.append(vel)
            self.sim_fbo.append(create_framebuffer(self.ctx, [pos, vel]))

    def resize(self, width: int, height: int) -> None:
        """Rebuilds the scene and bloom targets. Called on resize only."""
        w = max(1, int(width))
        h = max(1, int(height))
        if w == self.buffer_w and h == self.buffer_h:
            return

        self.buffer_w = w
        self.buffer_h = h
        self.aspect = w / h

        div = FX.bloom_divisor_low if self.low_power else FX.bloom_divisor
        self.bloom_w = max(1, w // div)
        self.bloom_h = max(1, h // div)

        for resource in [self.scene_tex, self.scene_fbo]:
            if resource is not None:
                resource.release()
        self._release_capture_targets()
        for t in self.bloom_tex:
            t.release()
        for f in self.bloom_fbo:
            f.release()
        self.bloom_tex = []
        self.bloom_fbo = []

        # Half float for the scene: the burst core runs well over 1.0 and 8-bit
        # would clip it flat before bloom ever sees it.
        self.scene_tex = create_texture(self.ctx, w, h, 4, "f2", None, moderngl.LINEAR)
        self.scene_fbo = create_framebuffer(self.ctx, [self.scene_tex])

        for _ in range(2):
            tex = create_texture(
                self.ctx, self.bloom_w, self.bloom_h, 4, "f2", None, moderngl.LINEAR
            )
            self.bloom_tex.append(tex)
            self.bloom_fbo.append(create_framebuffer(self.ctx, [tex]))

        if self.capture:
            self._allocate_capture_targets()

    def set_screen_rect(self, x: int, y: int, w: int, h: int) -> None:
        """
        Where the composite lands in the window.

        The browser build let CSS scale a smaller drawing buffer up to the
        viewport. Here the composite pass does it: it samples the scene target
        with linear filtering into whatever rectangle of the default framebuffer
        it is given, which is the whole window normally and a letterboxed slice
        while a clip is being framed.
        """
        self._screen_rect = (x, y, max(1, w), max(1, h))

    def set_capture(self, on: bool) -> None:
        """Routes the composite through an offscreen target so it can be read back."""
        self.capture = on
        if on:
            if self._present_fbo is None:
                self._allocate_capture_targets()
        else:
            self._release_capture_targets()

    def _allocate_capture_targets(self) -> None:
        self._present_tex = create_texture(
            self.ctx, self.buffer_w, self.buffer_h, 4, "f1", None, moderngl.LINEAR
        )
        self._present_fbo = create_framebuffer(self.ctx, [self._present_tex])
        size = self.buffer_w * self.buffer_h * 3
        self._readback_buffers = [
            self.ctx.buffer(reserve=size),
            self.ctx.buffer(reserve=size),
        ]
        self._readback_index = 0
        self._readback_ready = False

    def _release_capture_targets(self) -> None:
        for buffer in self._readback_buffers:
            buffer.release()
        self._readback_buffers = []
        self._readback_index = 0
        self._readback_ready = False
        if self._present_fbo is not None:
            self._present_fbo.release()
            self._present_fbo = None
        if self._present_tex is not None:
            self._present_tex.release()
            self._present_tex = None

    def prime_present_readback(self) -> None:
        """Queues the first GPU copy so the next capture can read it without a stall."""
        if (
            self._present_fbo is None
            or len(self._readback_buffers) != 2
            or self._readback_ready
        ):
            return
        self._present_fbo.read_into(
            self._readback_buffers[self._readback_index], components=3
        )
        self._readback_index ^= 1
        self._readback_ready = True

    def read_present(self) -> Optional[bytes]:
        """
        The previous composited frame at export resolution, bottom-up RGB.

        This readback is the one real cost the port adds: the browser recorded
        the canvas straight off its own compositor stream with no copy at all,
        and there is no equivalent of that outside a browser. It only runs while
        a clip is actually recording. Two pixel buffers pipeline the transfer so
        the CPU reads a completed copy while the GPU fills the other one.
        """
        if self._present_fbo is None or len(self._readback_buffers) != 2:
            return None
        if not self._readback_ready:
            self.prime_present_readback()
            return None

        write_index = self._readback_index
        self._present_fbo.read_into(self._readback_buffers[write_index], components=3)
        data = self._readback_buffers[write_index ^ 1].read()
        self._readback_index ^= 1
        return data

    def set_low_power(self, low: bool) -> None:
        """
        Flags the new tier and invalidates the targets. The caller follows with a
        resize(), because the pixel budget changes at the same time and rebuilding
        twice for one event is wasteful.
        """
        if low == self.low_power:
            return
        self.low_power = low
        self.buffer_w = 0

    def warmup(self) -> None:
        """
        Compiles and *runs* every shader variant before the first visible frame.

        Linking is not enough — plenty of drivers defer the real compile until a
        draw actually uses the program with a given uniform branch, and paying for
        that mid-session is a several-hundred-millisecond stall right when the
        first cast fires. So each element is pushed through the whole chain once
        while the loading screen is still up.
        """
        save_element = self.element
        save_word_alpha = self.word_alpha
        # Every value here has to clear the same early-outs the real frame uses,
        # or the pass gets skipped and the shader it was meant to warm is compiled
        # later — on the first cast, in front of somebody.
        for f in self.fx:
            f.active = True
            f.alpha = 0.02
            f.mode = MODE_BURST
        self.word_alpha = 0.02
        for el in (Element.FIRE, Element.ICE, Element.LIGHTNING):
            self.element = el
            self.bolts.generate(0.0, 0.0, 0.2, 2, 0.004)
            self._bolt_life = 1.0
            self.frame(1.0 / 60.0, 2.0)
        self.bolts.clear()
        self._bolt_life = 0.0
        for f in self.fx:
            f.active = False
            f.alpha = 0.0
            f.mode = MODE_CHARGE
        self.element = save_element
        self.word_alpha = save_word_alpha
        self.ctx.finish()

    # ----------------------------------------------------------- coordinates ---

    def set_video_aspect(self, video_aspect: float) -> None:
        # Cover fit: crop the long axis rather than letterbox.
        if video_aspect > self.aspect:
            self._cover_x = self.aspect / video_aspect
            self._cover_y = 1.0
        else:
            self._cover_x = 1.0
            self._cover_y = video_aspect / self.aspect

    def world_x_from_video(self, u: float) -> float:
        """
        Video-normalized (already mirrored) x to world x. The exact inverse of the
        uv mapping in VIDEO_VS — if one changes the other has to.
        """
        sign = 1.0 if self.mirror else -1.0
        return sign * ((2.0 * (u - 0.5)) / self._cover_x) * self.aspect * OVERSCAN

    def world_y_from_video(self, v: float) -> float:
        return ((2.0 * (0.5 - v)) / self._cover_y) * OVERSCAN

    @property
    def world_per_video_y(self) -> float:
        """World units per unit of normalized video y — used to size the rune."""
        return (2.0 / self._cover_y) * OVERSCAN

    # ---------------------------------------------------------------- events ---

    def respawn(self, slot: int, mode: int) -> None:
        """Re-seeds one hand's particles. Increments a float; uploads nothing."""
        f = self.fx[slot]
        # Wrapped well inside half-float integer range, in case the state
        # textures fell back to RGBA16F.
        f.mode = mode
        f.spawn_id = (f.spawn_id % 500) + 1

    def fire(self, slot: int, power: float, doubled: bool) -> None:
        """
        A cast. Everything expensive about this happened at init: the burst
        directions are already on the GPU, so all that changes here is a spawn id,
        a power uniform and a clock.
        """
        f = self.fx[slot]
        # A miss sputters. It must never be as good to look at as a hit, or the
        # wrong word becomes the more fun outcome.
        f.power = power * 0.25 if self.failing else power
        self.respawn(slot, MODE_BURST)

        combo = min(self.combo, FX.combo_cap) / FX.combo_cap
        strength = (0.5 + 0.5 * power) * (1.0 + combo * 0.5) * (1.55 if doubled else 1.0)
        self.view.pulse(strength * 0.25 if self.failing else strength)

        if self.element == Element.LIGHTNING and not self.failing:
            self._bolt_life = 1.0
            self._bolt_timer = 0.0
            self._bolt_x = f.x
            self._bolt_y = f.y
            self._bolt_reach = 0.75 + 0.55 * power + combo * 0.35
            self._bolt_power = power
            self.bolts.seed((slot * 7919 + round(self._time * 1000)) | 1)
            self._regenerate_bolts()

    def _regenerate_bolts(self) -> None:
        trunks = 9 + round(self._bolt_power * 5)
        self.bolts.generate(
            self._bolt_x, self._bolt_y, self._bolt_reach, trunks, 0.006 + self._bolt_power * 0.005
        )

    # ----------------------------------------------------------------- frame ---

    def frame(self, dt_sec: float, now_sec: float) -> None:
        ctx = self.ctx
        self._time = now_sec
        self.draw_calls = 0
        self.view.update(dt_sec)

        self._pack_uniforms()
        self._simulate(dt_sec)

        # --- scene ---
        self.scene_fbo.use()
        ctx.viewport = (0, 0, self.buffer_w, self.buffer_h)
        ctx.clear(0.043, 0.059, 0.102, 1.0)

        self._draw_video()

        ctx.enable(moderngl.BLEND)
        # Premultiplied additive. Everything glowing writes colour*alpha.
        ctx.blend_func = (moderngl.ONE, moderngl.ONE)
        self._draw_runes()
        self._draw_bolts(dt_sec)
        self._draw_particles()
        self._draw_word()
        ctx.disable(moderngl.BLEND)

        self._bloom()
        self._composite()

    def _pack_uniforms(self) -> None:
        combo = min(self.combo, FX.combo_cap) / FX.combo_cap
        base = ELEMENT_COLOR[int(self.element)]
        hot = ELEMENT_HOT[int(self.element)]

        if self.failing:
            # Magenta is the app's error colour and it is unmistakable against
            # all three elements.
            self._color[0], self._color[1], self._color[2] = FAIL_COLOR
            self._hot[0], self._hot[1], self._hot[2] = 1.0, 0.62, 0.8
        else:
            # Each consecutive correct cast shifts the palette hotter.
            for i in range(3):
                self._color[i] = base[i] + (hot[i] - base[i]) * combo * 0.55
                self._hot[i] = min(1.6, hot[i] * (1.0 + combo * 0.45))

        # COMPACTION, and it is load-bearing.
        #
        # The solver splits particles between two hands by column parity and
        # falls back to hand 0 whenever only one hand is up. So index 0 of every
        # per-hand uniform has to be a live hand. Writing them by raw slot index
        # means that a cast from the hand in slot 1, with slot 0 empty, sends
        # every particle to read an idle slot — and nothing renders at all.
        n = 0
        for s in range(2):
            f = self.fx[s]
            # A hand that has left the frame mid-burst keeps its slot until the
            # burst finishes, or its particles would be reassigned mid-flight.
            if not f.active and f.mode != MODE_BURST:
                continue
            self._u_palm[n * 3] = f.x
            self._u_palm[n * 3 + 1] = f.y
            self._u_palm[n * 3 + 2] = 0.0
            self._u_spawn_id[n] = f.spawn_id
            self._u_mode[n] = f.mode
            self._u_power[n] = f.power
            self._u_charge[n] = f.charge
            self._u_rune_pos[n * 2] = f.x
            self._u_rune_pos[n * 2 + 1] = f.y
            self._u_rune_size[n] = f.rune_size * (0.62 + 0.38 * f.charge)
            self._u_rune_alpha[n] = f.alpha if f.active else 0.0
            self._u_rune_fail[n] = f.fail
            n += 1
        self._packed = n

        for i in range(n, 2):
            self._u_palm[i * 3] = 0.0
            self._u_palm[i * 3 + 1] = 0.0
            self._u_palm[i * 3 + 2] = 0.0
            self._u_spawn_id[i] = 0.0
            self._u_mode[i] = MODE_CHARGE
            self._u_power[i] = 0.5
            self._u_charge[i] = 0.0
            self._u_rune_size[i] = 0.0
            self._u_rune_alpha[i] = 0.0
            self._u_rune_fail[i] = 0.0

    def _simulate(self, dt_sec: float) -> None:
        # A new charge/burst changes its spawn id, so waking from an idle state
        # reseeds every particle correctly. Until then, advancing two 128x128
        # float targets produces no pixels and is pure GPU/driver overhead.
        if not self._particles_active():
            return
        ctx = self.ctx
        src = self._ping_pong
        dst = src ^ 1

        self.sim_fbo[dst].use()
        ctx.viewport = (0, 0, self.tex_size, self.tex_size)
        ctx.disable(moderngl.BLEND)

        (
            self.p_sim.use()
            .tex("uPos", 0, self.pos_tex[src])
            .tex("uVel", 1, self.vel_tex[src])
            .tex("uSpawn", 2, self.spawn_tex)
            .float("uDt", min(1.0 / 30.0, dt_sec))
            .float("uTime", self._time)
            .int("uElement", int(self.element))
            .int("uHandCount", max(1, self._packed))
            .vec3v("uPalm", self._u_palm)
            .floatv("uSpawnId", self._u_spawn_id)
            .intv("uMode", self._u_mode)
            .floatv("uPower", self._u_power)
            .floatv("uCharge", self._u_charge)
            .float("uBurstLife", FX.burst_life)
        )
        self.tri.draw(self.p_sim)
        self.draw_calls += 1

        self._ping_pong = dst

    def _draw_video(self) -> None:
        self.ctx.disable(moderngl.BLEND)
        (
            self.p_video.use()
            .tex("uVideo", 0, self.video.texture)
            .mat3("uView", self.view.matrix)
            .float("uAspect", self.aspect)
            .float("uCamZ", CAM_Z)
            .vec2("uCoverScale", self._cover_x, self._cover_y)
            .float("uOverscan", OVERSCAN)
            .float("uMirror", 1.0 if self.mirror else -1.0)
            # Pulled down a touch so the effects sit on top of the person instead
            # of competing with a bright room. The face stays clearly readable,
            # which is the thing that decides whether a clip gets posted.
            .float("uDim", 0.9)
        )
        self._vao_video.render(moderngl.TRIANGLE_STRIP, vertices=4)
        self.draw_calls += 1

    def _draw_runes(self) -> None:
        if self._u_rune_alpha[0] < 0.004 and self._u_rune_alpha[1] < 0.004:
            return
        (
            self.p_rune.use()
            .mat3("uView", self.view.matrix)
            .float("uAspect", self.aspect)
            .float("uCamZ", CAM_Z)
            .float("uTime", self._time)
            .floatv("uCharge", self._u_charge)
            .floatv("uAlpha", self._u_rune_alpha)
            .floatv("uFail", self._u_rune_fail)
            .vec3("uColor", *self._color)
            .vec3("uHot", *self._hot)
            .vec3("uFailColor", *FAIL_COLOR)
            .float("uIntensity", self.intensity)
            # vec2[2] and float[2], written as raw bytes exactly as
            # uniform2fv/uniform1fv did.
            .floatv("uRunePos", self._u_rune_pos)
            .floatv("uRuneSize", self._u_rune_size)
        )
        self._vao_rune.render(moderngl.TRIANGLE_STRIP, vertices=4, instances=2)
        self.draw_calls += 1

    def _draw_bolts(self, dt_sec: float) -> None:
        if self._bolt_life <= 0:
            return
        self._bolt_life -= dt_sec / (FX.burst_life * 0.55)
        if self._bolt_life <= 0:
            self._bolt_life = 0.0
            self.bolts.clear()
            return

        self._bolt_timer -= dt_sec * 1000.0
        if self._bolt_timer <= 0:
            self._bolt_timer = FX.bolt_regen_ms
            self._regenerate_bolts()

        (
            self.p_bolt.use()
            .mat3("uView", self.view.matrix)
            .float("uAspect", self.aspect)
            .float("uCamZ", CAM_Z)
            .vec3("uColor", *self._color)
            .vec3("uHot", *self._hot)
            .float("uFade", self._bolt_life * self._bolt_life)
            .float("uIntensity", self.intensity)
        )
        self.bolts.draw()
        self.draw_calls += 1

    def _draw_particles(self) -> None:
        # Nothing charging and nothing burning: skip the draw entirely rather
        # than pay 16,384 instances to render nothing.
        if not self._particles_active():
            self.particles_drawn = 0
            return
        bursting = self._u_mode[0] == MODE_BURST or self._u_mode[1] == MODE_BURST
        full = self.max_particles >> 1 if self.low_power else self.max_particles
        charge = 1400 if self.low_power else 2800
        count = (full // 6 if self.failing else full) if bursting else charge
        self.particles_drawn = count

        combo = min(self.combo, FX.combo_cap) / FX.combo_cap
        (
            self.p_draw.use()
            .tex("uPos", 0, self.pos_tex[self._ping_pong])
            .tex("uVel", 1, self.vel_tex[self._ping_pong])
            .tex("uSpawn", 2, self.spawn_tex)
            .mat3("uView", self.view.matrix)
            .float("uAspect", self.aspect)
            .float("uCamZ", CAM_Z)
            .float("uTime", self._time)
            .float("uSize", 0.0062 * (1.0 + combo * 0.5))
            .int("uElement", int(self.element))
            .int("uHandCount", max(1, self._packed))
            .intv("uMode", self._u_mode)
            .floatv("uChargeAlpha", self._u_rune_alpha)
            .vec3("uColor", *self._color)
            .vec3("uHot", *self._hot)
            .float("uIntensity", self.intensity)
        )
        self._vao_particles.render(moderngl.TRIANGLE_STRIP, vertices=4, instances=count)
        self.draw_calls += 1

    def _particles_active(self) -> bool:
        return bool(
            self._u_mode[0] == MODE_BURST
            or self._u_mode[1] == MODE_BURST
            or self._u_rune_alpha[0] >= 0.004
            or self._u_rune_alpha[1] >= 0.004
        )

    def _draw_word(self) -> None:
        if self.word_alpha < 0.004:
            return
        # Sized to span most of the frame width, then clamped so a long word does
        # not run off the sides.
        half_w = min(self.aspect * 0.92, 1.5 * self.word_tex.aspect) * self.word_scale
        half_h = half_w / max(0.5, self.word_tex.aspect)
        (
            self.p_word.use()
            .tex("uWord", 0, self.word_tex.texture)
            .mat3("uView", self.view.matrix)
            .float("uAspect", self.aspect)
            .float("uCamZ", CAM_Z)
            .vec2("uWordCenter", 0.0, 0.0)
            .vec2("uWordSize", half_w, half_h)
            .float("uDissolve", self.word_dissolve)
            .float("uAlpha", self.word_alpha)
            .vec3("uColor", *self._color)
            .vec3("uHot", *self._hot)
            .float("uIntensity", self.intensity)
        )
        self._vao_word.render(moderngl.TRIANGLE_STRIP, vertices=4)
        self.draw_calls += 1

    def _bloom(self) -> None:
        ctx = self.ctx
        ctx.disable(moderngl.BLEND)

        self.bloom_fbo[0].use()
        ctx.viewport = (0, 0, self.bloom_w, self.bloom_h)
        (
            self.p_bloom_down.use()
            .tex("uScene", 0, self.scene_tex)
            .vec2("uTexel", 1.0 / self.buffer_w, 1.0 / self.buffer_h)
            .float("uThreshold", 0.62)
            .float("uKnee", 0.35)
        )
        self.tri.draw(self.p_bloom_down)
        self.draw_calls += 1

        self.bloom_fbo[1].use()
        ctx.viewport = (0, 0, self.bloom_w, self.bloom_h)
        self.p_bloom_blur.use().tex("uSrc", 0, self.bloom_tex[0]).vec2("uDir", 1.0 / self.bloom_w, 0.0)
        self.tri.draw(self.p_bloom_blur)
        self.draw_calls += 1

        self.bloom_fbo[0].use()
        ctx.viewport = (0, 0, self.bloom_w, self.bloom_h)
        self.p_bloom_blur.use().tex("uSrc", 0, self.bloom_tex[1]).vec2("uDir", 0.0, 1.0 / self.bloom_h)
        self.tri.draw(self.p_bloom_blur)
        self.draw_calls += 1

    def _composite(self) -> None:
        ctx = self.ctx
        if self.capture and self._present_fbo is not None:
            self._present_fbo.use()
            ctx.viewport = (0, 0, self.buffer_w, self.buffer_h)
        else:
            ctx.screen.use()
            ctx.viewport = self._screen_rect

        (
            self.p_composite.use()
            .tex("uScene", 0, self.scene_tex)
            .tex("uBloom", 1, self.bloom_tex[0])
            .float("uBloomAmount", (0.9 if self.low_power else 1.15) * self.intensity)
            .float("uAberration", self.view.aberration * 0.02)
            .float("uTime", self._time)
            .float("uVignette", 0.55)
            .float("uFlash", self.view.flash * 0.35)
            .vec3(
                "uFlashColor",
                FAIL_COLOR[0] if self.failing else float(self._hot[0]),
                FAIL_COLOR[1] if self.failing else float(self._hot[1]),
                FAIL_COLOR[2] if self.failing else float(self._hot[2]),
            )
        )
        self.tri.draw(self.p_composite)
        self.draw_calls += 1

        if self.capture and self._present_tex is not None:
            # One extra pass, and only while recording: put the export-resolution
            # frame on the screen inside its letterbox.
            ctx.screen.use()
            screen_w, screen_h = self.window.framebuffer_size
            ctx.viewport = (0, 0, screen_w, screen_h)
            ctx.clear(0.0, 0.0, 0.0, 1.0)
            ctx.viewport = self._screen_rect
            self.p_blit.use().tex("uSrc", 0, self._present_tex)
            self.tri.draw(self.p_blit)
            self.draw_calls += 1

    def begin_overlay(self) -> None:
        """
        Hands the default framebuffer to the 2D chrome, with the viewport back at
        the full window. The overlay is not part of a recording, exactly as the
        browser's DOM was not part of a canvas captureStream.
        """
        self.ctx.screen.use()
        screen_w, screen_h = self.window.framebuffer_size
        self.ctx.viewport = (0, 0, screen_w, screen_h)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA)

    def end_overlay(self) -> None:
        self.ctx.disable(moderngl.BLEND)
