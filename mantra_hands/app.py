"""
Orchestration: the two clocks, cast resolution, framing.

The browser build ran detection inside ``requestVideoFrameCallback`` and
everything else inside ``requestAnimationFrame``. Here detection has a thread of
its own, blocked on the camera, and everything else runs on the main thread,
blocked on the display's vsync. The recognizers still run on the detection clock,
reading measured landmarks at camera rate — that is the part that matters. What
they emit is handed to the main thread through a queue and applied at the top of
the next frame, at most one frame later, because effects and scoring touch GL and
GL belongs to one thread.
"""

from __future__ import annotations

import logging
import math
import queue
from dataclasses import dataclass
from typing import Optional

import glfw
import moderngl

from .audio.speech import SpeechScorer, create_scorer, speech_supported
from .camera import Camera, CameraError
from .clock import now_ms
from .config import ELEMENT_LABEL, FX, GESTURE, RECORD, VIDEO
from .content.deck import Deck, WordEntry, load_deck
from .demo import DemoCamera, DemoHands
from .gestures import CastRecognizer, Phase
from .perf import Perf
from .render.gl import Window, WindowError
from .render.renderer import MODE_BURST, MODE_CHARGE, Renderer
from .tracking import Tracker
from .ui.hud import Hud, HudInput
from .ui.progress import Progress, Settings
from .ui.recorder import Recorder
from .ui.shell import Shell

log = logging.getLogger("mantra_hands")

WORD_SHOW_SEC = 1.25
NEXT_WORD_DELAY_MS = 1150


@dataclass
class SlotFx:
    """Per-hand presentation state that is not part of the recognizer."""

    #: 0..1, the rune blowing outward after a cast.
    pop: float = 0.0
    #: 0..1, the rune cracking after a miss.
    fail: float = 0.0
    #: Seconds since this hand's burst started. -1 when it is not bursting.
    burst_age: float = -1.0


class App:
    def __init__(
        self,
        demo: bool = False,
        deck_path: Optional[str] = None,
        camera_index: int = 0,
        fullscreen: bool = False,
        width: int = 1280,
        height: int = 720,
        visible: bool = True,
    ) -> None:
        self.demo_mode = demo
        self.deck_path = deck_path
        self.camera_index = camera_index

        # `visible=False` runs the entire app into an off-screen framebuffer,
        # which is how the render path gets verified without a display session.
        self.window = Window(width, height, "Mantra Hands", fullscreen=fullscreen, visible=visible)
        self.window.on_key = self._on_key
        self.window.on_resize = self._on_resize

        self.progress = Progress()
        self.shell = Shell(self.window.ctx, self.progress.settings)
        self.perf = Perf()
        self.hud = Hud(self.window.ctx)
        self.tracker = Tracker()
        self.recognizers = [CastRecognizer(), CastRecognizer()]
        self.slots = [SlotFx(), SlotFx()]

        self.renderer: Optional[Renderer] = None
        self.recorder = Recorder()
        self.deck: Optional[Deck] = None
        self.scorer: SpeechScorer = create_scorer(False)
        self.camera = None

        self.word: Optional[WordEntry] = None
        self.pending_word: Optional[WordEntry] = None
        self.combo = 0
        self.state = "explainer"
        self.framed = False
        self.last_cast_ms = -1.0
        self.next_word_at = -1.0
        self.word_t = -1.0
        self.missing_since = -1.0
        self.video_aspect = 4.0 / 3.0
        self.charge_owner = -1

        self._events: "queue.SimpleQueue[tuple]" = queue.SimpleQueue()
        self._pending_boot: Optional[bool] = None

        w, h = self.window.framebuffer_size
        self.shell.resize(w, h)
        self.hud.resize(w, h)

        self.recorder.set_framing = self._set_framing
        self.recorder.on_message = lambda msg: self.shell.toast(msg, now_ms(), True)

    # ------------------------------------------------------------------ boot ---

    def present(self) -> None:
        """Shown before the operating system's own camera prompt."""
        if self.demo_mode:
            self._pending_boot = False
            return
        note = (
            "Everything runs on this device. Nothing is uploaded."
            if speech_supported()
            else "No offline speech model found, so the app will score you offline instead. "
            "Fully playable."
        )
        self.shell.show_explainer(note)

    def boot(self, with_mic: bool) -> None:
        self.shell.hide_explainer()
        self._loading("Starting demo mode" if self.demo_mode else "Opening the camera")

        try:
            if self.demo_mode:
                self.tracker.demo = DemoHands()
                self.camera = DemoCamera(VIDEO.width, VIDEO.height, VIDEO.fps)
            else:
                self.camera = Camera(self.camera_index)
            self.video_aspect = self.camera.aspect
            self.tracker.set_video_aspect(self.video_aspect)

            self._loading("Warming up the graphics")
            self.renderer = Renderer(self.window)
            self.apply_canvas_size()

            self._loading("Loading words")
            self.deck = load_deck(self.deck_path)
            self.shell.set_deck_info(
                f"{self.deck.size} words from {self.deck.name} ({self.deck.source})."
            )

            self.tracker.load(self._loading)

            # Every shader variant compiled and drawn once while the loading
            # screen is still up. A shader compiling mid-session is a several
            # hundred millisecond stall, and it would land on the first cast.
            self._loading("Compiling spells")
            self.renderer.warmup()

            self.start_scorer(with_mic)
            self.attach_governor()
            self.apply_settings(self.progress.settings)
            self.next_word()

            self.tracker.on_detect = self.on_detect
            self.tracker.start(self.camera)

            self.shell.hide_loading()
            self.shell.set_counts(0, self.progress.streak, self.progress.best)
            if self.demo_mode:
                self.shell.toast("Demo mode — scripted hands, no camera.", now_ms())
            self.state = "running"
        except Exception as err:  # noqa: BLE001
            log.exception("startup failed")
            message = str(err) if isinstance(err, (CameraError, WindowError)) else (
                f"{type(err).__name__}: {err}"
            )
            self.shell.show_fatal(message or "Something went wrong starting up.")
            self.state = "fatal"

    def _loading(self, message: str) -> None:
        """
        Puts a message up and paints one frame before the caller blocks.

        Opening a camera and loading an 8 MB model both take long enough that a
        screen with nothing on it reads as a hang.
        """
        self.shell.show_loading(message)
        self._paint_chrome()

    def start_scorer(self, with_mic: bool) -> None:
        want_native = with_mic and self.progress.settings.mic is not False
        self.scorer = create_scorer(want_native)
        self.scorer.on_heard = lambda text, correct: self.shell.show_heard(text, correct, now_ms())

        def trouble(reason: str) -> None:
            if reason in ("network", "service-not-allowed", "no model"):
                self.shell.toast("Speech engine unavailable — scoring offline now.", now_ms(), True)
                self.swap_scorer(False)
            elif reason == "not-allowed":
                self.shell.toast("Microphone blocked — scoring offline instead.", now_ms(), True)
                self.swap_scorer(False)

        self.scorer.on_trouble = trouble
        if with_mic:
            self.scorer.begin()

        using_mic = self.scorer.kind == "web"
        self.shell.set_mic_enabled(using_mic)
        self.progress.settings.mic = using_mic
        self.shell.set_mic_note(
            "Listening with the offline speech model on this machine."
            if using_mic
            else "Off. The app scores you offline — fully playable, no mic needed."
            if speech_supported()
            else "No offline speech model installed, so scoring is offline."
        )

    def swap_scorer(self, with_mic: bool) -> None:
        self.scorer.end()
        self.start_scorer(with_mic)
        if self.word:
            self.scorer.listen_for(self.word.word)

    def apply_settings(self, s: Settings) -> None:
        if self.renderer is None:
            return
        self.renderer.intensity = s.intensity / 100.0
        # The mirror is a display decision. The landmark pipeline stays mirrored
        # either way; the renderer flips the camera frame and the palm-to-world
        # mapping together so they can never disagree.
        self.renderer.mirror = s.mirror
        if s.low_power != self.perf.low_power:
            self.perf.set_low_power(s.low_power)
            self.renderer.set_low_power(s.low_power)
            self.window.set_swap_interval(2 if s.low_power else 1)
            self.apply_canvas_size()
        if s.mic != (self.scorer.kind == "web"):
            self.swap_scorer(s.mic)
        self.progress.save()

    def attach_governor(self) -> None:
        """
        Two seconds over budget and the app steps itself down — and says so. A
        silent quality drop just reads as the app getting worse for no reason.
        """

        def degrade() -> None:
            self.renderer.set_low_power(True)
            self.window.set_swap_interval(2)
            self.apply_canvas_size()
            self.shell.set_low_power_toggle(True)
            self.progress.save()
            self.shell.toast("Switched to low power mode to keep things smooth.", now_ms(), True)

        self.perf.on_low_power = degrade

    # -------------------------------------------------------------- detection ---

    def on_detect(self, t_ms: float) -> None:
        """
        Runs on the detection clock, at camera rate. Never inside the render loop.
        """
        for s in range(2):
            rec = self.recognizers[s]
            rec.update(self.tracker.hands[s], t_ms)
            if rec.started_charging:
                self._events.put(("charge", s))
            if rec.fired:
                other = self.recognizers[1 - s]
                # Both hands pushing right now, with the other a frame behind.
                # Judged here, on the detection clock, because a frame later the
                # other hand's rate has already decayed.
                together = (
                    other.armed
                    and other.expand_rate > GESTURE.expand_rate_on * 0.55
                    and self.tracker.hands_visible > 1
                )
                self._events.put(("fire", s, t_ms, rec.power, together))

    def _drain_events(self) -> None:
        while True:
            try:
                event = self._events.get_nowait()
            except queue.Empty:
                break
            if event[0] == "charge":
                self.on_charge_start(event[1])
            else:
                _, slot, t_ms, power, together = event
                self.on_fire(slot, t_ms, power, together)
        if self._charging_slots == 0:
            self.charge_owner = -1

    @property
    def _charging_slots(self) -> int:
        return sum(1 for r in self.recognizers if r.phase in (Phase.CHARGING, Phase.ARMED))

    def on_charge_start(self, slot: int) -> None:
        # Proves the palm-facing sign is right, so the tracker stops second
        # guessing MediaPipe's handedness convention.
        self.tracker.confirm_handedness()
        self.renderer.respawn(slot, MODE_CHARGE)
        self.renderer.failing = False
        self.slots[slot].fail = 0.0
        self.try_swap_element(True)

        # Only the first hand to charge opens the listening window; the second
        # hand joining must not wipe what has already been heard.
        if self.charge_owner < 0:
            self.charge_owner = slot
            self.shell.clear_heard()
            if self.word:
                self.scorer.listen_for(self.word.word)

    def on_fire(self, slot: int, t_ms: float, power: float, together: bool) -> None:
        fx = self.renderer.fx[slot]

        # The second hand of a double cast, arriving a frame or two after the
        # first. Add its power to the cast already in flight rather than starting
        # a new one — and do not judge the word twice.
        if self.last_cast_ms >= 0 and t_ms - self.last_cast_ms < GESTURE.double_cast_window_ms:
            self.renderer.fire(slot, power, True)
            self.slots[slot].pop = 1.0
            self.slots[slot].fail = 1.0 if self.renderer.failing else 0.0
            self.slots[slot].burst_age = 0.0
            fx.alpha = 1.0
            return

        result = self.scorer.check()
        correct = result.correct
        self.renderer.failing = not correct
        self.renderer.fire(slot, power, together)
        self.last_cast_ms = t_ms
        self.slots[slot].pop = 1.0 if correct else 0.4
        self.slots[slot].fail = 0.0 if correct else 1.0
        self.slots[slot].burst_age = 0.0
        fx.alpha = 1.0

        if correct:
            self.succeed(together)
        else:
            self.miss(result.heard)

    def succeed(self, doubled: bool) -> None:
        self.combo += 1
        record = self.progress.correct()
        self.renderer.combo = self.combo
        self.shell.set_counts(self.combo, self.progress.streak, self.progress.best)
        self.word_t = 0.0
        self.next_word_at = now_ms() + NEXT_WORD_DELAY_MS
        if record and self.progress.best >= 3:
            self.shell.toast(f"New best — {self.progress.best}!", now_ms())
        elif doubled:
            self.shell.toast("Double cast. Mantap!", now_ms())
        elif self.combo == 5:
            self.shell.toast("Five in a row. Keep going!", now_ms())

    def miss(self, heard: bool) -> None:
        self.combo = 0
        self.progress.wrong()
        self.renderer.combo = 0
        self.shell.set_counts(0, self.progress.streak, self.progress.best)
        line = self.shell.fizzle()
        self.shell.toast(
            line if heard else "Nothing heard — say it out loud while you charge.", now_ms()
        )
        # The word stays up. A miss is a retry, not a new question.

    def next_word(self) -> None:
        """
        The prompt advances straight away; the renderer catches up later.

        The two cannot move together. A cast's burst outlives the pause before the
        next word by a good half second, and switching the element uniform or the
        word texture underneath it would recolour a fire burst to ice mid-flight
        and swap the word out from under its own dissolve.
        """
        self.word = self.deck.next()
        self.shell.set_word(self.word)
        # Upload into the inactive word texture now, while the previous burst is
        # still finishing. If the user charges early, switching is pointer-only.
        self.renderer.word_tex.prepare(self.word.word)
        self.pending_word = self.word
        self.next_word_at = -1.0
        self.try_swap_element()

    def try_swap_element(self, force: bool = False) -> None:
        """
        :param force: set when a charge begins. The rune the user is charging has
            to be the element of the word they are about to say, even if that
            means recolouring the tail of the previous burst.
        """
        pending = self.pending_word
        if pending is None:
            return
        if not force:
            if self.renderer.word_alpha > 0.004:
                return
            for slot in self.slots:
                if 0 <= slot.burst_age < FX.burst_life * 1.1:
                    return
        self.renderer.element = pending.element
        self.renderer.word_tex.activate(pending.word)
        self.renderer.failing = False
        self.pending_word = None

    # ------------------------------------------------------------------ frame ---

    def run(self) -> None:
        self.present()
        try:
            while not self.window.should_close:
                self.step()
        finally:
            self.destroy()

    def step(self) -> None:
        """One iteration of the main loop. Split out so the whole app can be
        driven a frame at a time by tests/smoke_render.py."""
        self.window.poll()
        if self._pending_boot is not None:
            with_mic = self._pending_boot
            self._pending_boot = None
            self.boot(with_mic)
        elif self.state == "running":
            self._frame()
        else:
            self._paint_chrome()

    def _frame(self) -> None:
        now = now_ms()
        dt_ms = self.perf.begin_frame(now)
        dt = min(0.05, dt_ms / 1000.0)

        # The detection thread left a frame; hand it to the GPU. Uploading here
        # rather than every rendered frame skips the duplicate upload on the ~50%
        # of display frames with no new image.
        self.renderer.video.upload(self.tracker.take_frame())

        # Landmarks stepped forward from the last detection to this display frame.
        self.tracker.predict(now)
        self._drain_events()
        self.update_fx(dt, now)

        self.renderer.frame(dt, now / 1000.0)
        self.recorder.update(now, self.renderer)
        self.shell.set_countdown(self.recorder.countdown_text)

        self.renderer.begin_overlay()
        self.renderer.draw_calls += self.shell.draw(now)
        self.renderer.draw_calls += self.hud.draw()
        self.renderer.end_overlay()

        self.perf.detect_ms = self.tracker.detect_ms
        self.perf.detect_hz = self.tracker.detect_hz
        self.perf.end_frame(now_ms())
        self.update_hud(now)

        self.window.swap()

        if not self.tracker.running and self.state == "running":
            self.shell.show_fatal("The camera stopped sending video. Reconnect it and restart.")
            self.state = "fatal"

    def _paint_chrome(self) -> None:
        """The explainer, the loading screen and the fatal sheet, with no scene."""
        ctx = self.window.ctx
        ctx.screen.use()
        w, h = self.window.framebuffer_size
        ctx.viewport = (0, 0, w, h)
        ctx.clear(0.043, 0.059, 0.102, 1.0)
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = (moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA)
        self.shell.draw(now_ms())
        ctx.disable(moderngl.BLEND)
        self.window.swap()

    def update_fx(self, dt: float, now: float) -> None:
        renderer = self.renderer
        any_hand = False
        charge_shown = 0.0
        armed_shown = False

        for s in range(2):
            hand = self.tracker.hands[s]
            rec = self.recognizers[s]
            fx = renderer.fx[s]
            slot = self.slots[s]

            fx.active = hand.present
            if hand.present:
                any_hand = True
                fx.x = renderer.world_x_from_video(hand.view_palm_x)
                fx.y = renderer.world_y_from_video(hand.view_palm_y)
                if rec.charge01 > charge_shown:
                    charge_shown = rec.charge01
                if rec.armed:
                    armed_shown = True

            charging = hand.present and rec.phase in (Phase.CHARGING, Phase.ARMED)
            if charging:
                fx.alpha += (1.0 - fx.alpha) * (1.0 - math.exp(-dt * 14.0))
                slot.pop *= math.exp(-dt * 9.0)
                slot.fail *= math.exp(-dt * 5.0)
            else:
                fx.alpha *= math.exp(-dt * 6.0)
                slot.pop *= math.exp(-dt * 5.5)
                slot.fail *= math.exp(-dt * 2.6)
            if fx.alpha < 0.003:
                fx.alpha = 0.0

            fx.charge = rec.charge01
            fx.fail = slot.fail
            # The rune sits on the palm and is sized from the hand, so it stays
            # put as the user moves toward or away from the camera. On a cast it
            # blows outward; on a miss it barely swells and cracks instead.
            base = hand.hand_size * renderer.world_per_video_y * 1.5
            fx.rune_size = base * (1.0 + slot.pop * 1.25)

            if fx.mode == MODE_BURST:
                # Power keeps being revised for a moment after the cast, so the
                # burst swells to match how hard the push turned out to be.
                fx.power = max(fx.power, rec.power)
                slot.burst_age += dt
                # Once the burst has burned out the same particles go back to
                # charge duty, which also drops the draw from 16k instances
                # to 2.8k.
                spent = slot.burst_age > FX.burst_life * 1.6
                if spent and (charging or fx.alpha == 0.0):
                    slot.burst_age = -1.0
                    renderer.respawn(s, MODE_CHARGE)

        # The huge word across the burst, then a dissolve.
        if self.word_t >= 0:
            self.word_t += dt
            t = self.word_t / WORD_SHOW_SEC
            renderer.word_alpha = t / 0.07 if t < 0.07 else 1.0
            renderer.word_dissolve = max(0.0, (t - 0.22) / 0.78)
            renderer.word_scale = 1.0 + t * 0.14
            if t >= 1.0:
                self.word_t = -1.0
                renderer.word_alpha = 0.0

        if self.next_word_at > 0 and now >= self.next_word_at:
            self.next_word()
        self.try_swap_element()

        # "Show me your hand" — never a blank screen, but not a flicker either.
        if any_hand:
            self.missing_since = -1.0
        elif self.missing_since < 0:
            self.missing_since = now
        self.shell.set_empty_state(
            not any_hand and self.missing_since > 0 and now - self.missing_since > 450
        )

        self.shell.set_charge(charge_shown, armed_shown)
        self.shell.set_hint(
            "PUSH!"
            if armed_shown
            else "Hold it — say the word out loud"
            if charge_shown > 0
            else "Open palm · say the word · push"
        )

    def update_hud(self, now: float) -> None:
        if not self.hud.visible:
            return
        rec = self.recognizers[0] if self.recognizers[0].phase != Phase.IDLE else self.recognizers[1]
        self.hud.update(
            self.perf,
            HudInput(
                draw_calls=self.renderer.draw_calls,
                particles=self.renderer.particles_drawn,
                hands=self.tracker.hands_visible,
                delegate=self.tracker.delegate,
                phase=rec.phase_name,
                element=ELEMENT_LABEL[int(self.renderer.element)],
                combo=self.combo,
                expand_rate=rec.expand_rate,
                facing=rec.facing,
                renderer=self.window.caps.renderer,
                handedness=self.tracker.handedness,
                scorer=self.scorer.kind,
            ),
            now,
        )

    # ----------------------------------------------------------------- chrome ---

    def apply_canvas_size(self) -> None:
        if self.renderer is None:
            return
        screen_w, screen_h = self.window.framebuffer_size
        if self.framed:
            w = RECORD.width_low if self.perf.low_power else RECORD.width
            h = RECORD.height_low if self.perf.low_power else RECORD.height
            self.renderer.resize(w, h)
            # Letterboxed, so the framing during the countdown is exactly the
            # framing that gets exported.
            scale = min(screen_w / w, screen_h / h)
            rw, rh = int(w * scale), int(h * scale)
            self.renderer.set_screen_rect((screen_w - rw) // 2, (screen_h - rh) // 2, rw, rh)
        else:
            # Budget by pixel count. A 4K panel would otherwise ask for 8.3
            # million pixels a frame, and the burst is fill-rate bound long
            # before it is anything else bound.
            budget = 1_100_000 if self.perf.low_power else 2_100_000
            scale = max(0.6, min(1.0, math.sqrt(budget / max(1, screen_w * screen_h))))
            self.renderer.resize(round(screen_w * scale), round(screen_h * scale))
            self.renderer.set_screen_rect(0, 0, screen_w, screen_h)
        self.renderer.set_video_aspect(self.video_aspect)

    def _set_framing(self, on: bool) -> None:
        self.framed = on
        if self.renderer is not None:
            # Drop any readback target before the resize releases it, then ask
            # for one at the new size. One reallocation, not a render pass.
            self.renderer.set_capture(False)
            self.apply_canvas_size()
            self.renderer.set_capture(on)
        self.shell.set_recording(on)

    def _on_resize(self, width: int, height: int) -> None:
        self.shell.resize(width, height)
        self.hud.resize(width, height)
        # Never re-lay-out mid recording; the encoder is fed a fixed frame size.
        if self.framed:
            return
        self.apply_canvas_size()

    def _on_key(self, key: int, mods: int) -> None:
        if self.state == "explainer":
            if key in (glfw.KEY_ENTER, glfw.KEY_KP_ENTER):
                self._pending_boot = True
            elif key == glfw.KEY_N:
                self._pending_boot = False
            elif key in (glfw.KEY_Q, glfw.KEY_ESCAPE):
                self.window.request_close()
            return

        if key == glfw.KEY_Q:
            self.window.request_close()
            return
        if key == glfw.KEY_D:
            self.hud.toggle()
            return
        if self.state != "running":
            return

        settings = self.progress.settings
        if key == glfw.KEY_ESCAPE:
            self.shell.toggle_settings(False)
        elif key == glfw.KEY_S:
            self.shell.toggle_settings(not self.shell.settings_open)
        elif key == glfw.KEY_M:
            settings.mirror = not settings.mirror
            self.apply_settings(settings)
            self.shell.refresh()
        elif key == glfw.KEY_P:
            settings.low_power = not settings.low_power
            self.apply_settings(settings)
            self.shell.refresh()
        elif key in (glfw.KEY_LEFT_BRACKET, glfw.KEY_RIGHT_BRACKET):
            step = 5 if key == glfw.KEY_RIGHT_BRACKET else -5
            settings.intensity = max(40, min(150, settings.intensity + step))
            self.apply_settings(settings)
            self.shell.refresh()
        elif key == glfw.KEY_R:
            self.recorder.toggle(now_ms())

    def destroy(self) -> None:
        self.state = "stopped"
        self.tracker.stop()
        if self.scorer:
            self.scorer.end()
        if self.camera is not None:
            self.camera.release()
        self.recorder.close()
        self.progress.save()
        self.window.destroy()
