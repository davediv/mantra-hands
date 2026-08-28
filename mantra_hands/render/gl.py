"""
Thin GL helpers. Everything in here runs at init. Nothing allocates once the
frame loop has started.

The browser build asked a canvas for a ``webgl2`` context. This asks GLFW for a
3.3 core context — the desktop profile that maps one-to-one onto WebGL2 — and
wraps it in moderngl.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional, Sequence, Tuple

import glfw
import moderngl
import numpy as np

from .shaders.common import VERSION


@dataclass
class GLCaps:
    #: Renderable 32-bit float colour. The particle solver wants it.
    color_float: bool
    color_half_float: bool
    renderer: str
    version: str
    max_texture_size: int


class WindowError(RuntimeError):
    """Raised with copy that is safe to put in front of a user."""


class Window:
    """
    The window, its GL context, and the keyboard.

    Vsync is on, so ``swap()`` blocks until the display is ready for the next
    frame. That is the main loop's clock, exactly as ``requestAnimationFrame``
    was in the browser.
    """

    def __init__(
        self,
        width: int,
        height: int,
        title: str,
        fullscreen: bool = False,
        visible: bool = True,
    ) -> None:
        if not glfw.init():
            raise WindowError("Could not initialise GLFW — no display available.")

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
        # Everything lands in an offscreen HDR target and comes back through one
        # composite pass, so the default framebuffer never needs MSAA or depth.
        glfw.window_hint(glfw.SAMPLES, 0)
        glfw.window_hint(glfw.DEPTH_BITS, 0)
        glfw.window_hint(glfw.STENCIL_BITS, 0)
        # Hidden windows are how the render path gets verified without a display
        # session; see tests/smoke_render.py.
        glfw.window_hint(glfw.VISIBLE, visible)

        monitor = glfw.get_primary_monitor() if fullscreen else None
        if fullscreen and monitor:
            mode = glfw.get_video_mode(monitor)
            width, height = mode.size.width, mode.size.height

        self.handle = glfw.create_window(width, height, title, monitor, None)
        if not self.handle:
            glfw.terminate()
            raise WindowError(
                "Could not open a window with an OpenGL 3.3 core context. "
                "This machine's driver may be too old for the particle solver."
            )

        glfw.make_context_current(self.handle)
        glfw.swap_interval(1)
        self.ctx = moderngl.create_context()

        info = self.ctx.info
        self.caps = GLCaps(
            color_float=True,  # RGBA32F render targets are core in GL 3.0+
            color_half_float=True,
            renderer=str(info.get("GL_RENDERER", "?")),
            version=str(info.get("GL_VERSION", "?")),
            max_texture_size=int(info.get("GL_MAX_TEXTURE_SIZE", 4096)),
        )

        self.on_key: Optional[Callable[[int, int], None]] = None
        self.on_resize: Optional[Callable[[int, int], None]] = None
        glfw.set_key_callback(self.handle, self._key_callback)
        glfw.set_framebuffer_size_callback(self.handle, self._size_callback)

        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.CULL_FACE)

    def _key_callback(self, window, key, scancode, action, mods) -> None:  # noqa: ANN001
        if action == glfw.PRESS and self.on_key:
            self.on_key(key, mods)

    def _size_callback(self, window, width, height) -> None:  # noqa: ANN001
        if self.on_resize:
            self.on_resize(max(1, width), max(1, height))

    def set_swap_interval(self, interval: int) -> None:
        """
        How many display refreshes a swap waits for.

        This is the desktop form of the browser build's 30 fps cap in low power
        mode. Skipping every other frame in software would either spin the CPU or
        present a stale buffer; asking the driver to hold the swap does neither,
        and it keeps the skipped time out of the frame-time average so the HUD
        reports 30 fps rather than a bimodal mess.
        """
        glfw.swap_interval(max(1, interval))

    @property
    def framebuffer_size(self) -> Tuple[int, int]:
        """
        Pixels, not points. On a Retina display these differ by the scale factor,
        and it is the pixel count that the fill-rate budget cares about.
        """
        w, h = glfw.get_framebuffer_size(self.handle)
        return max(1, w), max(1, h)

    @property
    def should_close(self) -> bool:
        return bool(glfw.window_should_close(self.handle))

    def request_close(self) -> None:
        glfw.set_window_should_close(self.handle, True)

    def poll(self) -> None:
        glfw.poll_events()

    def swap(self) -> None:
        glfw.swap_buffers(self.handle)

    def destroy(self) -> None:
        glfw.destroy_window(self.handle)
        glfw.terminate()


class Program:
    """
    A linked program with its uniform lookups cached, including the misses.

    A uniform the compiler dropped because nothing read it is not an error — the
    browser build got ``null`` back from ``getUniformLocation`` and carried on,
    and so does this. Every program in the app is built during init; the frame
    loop never compiles.
    """

    def __init__(self, ctx: moderngl.Context, vertex_src: str, fragment_src: str, label: str) -> None:
        self.ctx = ctx
        self.label = label
        try:
            self.handle = ctx.program(vertex_shader=vertex_src, fragment_shader=fragment_src)
        except Exception as err:  # noqa: BLE001
            raise WindowError(f"{label} failed to compile:\n{err}") from err
        self._locs: Dict[str, Optional[moderngl.Uniform]] = {}

    def loc(self, name: str) -> Optional[moderngl.Uniform]:
        if name not in self._locs:
            try:
                self._locs[name] = self.handle[name]
            except KeyError:
                self._locs[name] = None
        return self._locs[name]

    def use(self) -> "Program":
        """
        Present for parity with the browser build's call sites. moderngl binds
        the program through the vertex array, so this only has to exist.
        """
        return self

    def int(self, name: str, v: int) -> "Program":
        u = self.loc(name)
        if u is not None:
            u.value = int(v)
        return self

    def float(self, name: str, v: float) -> "Program":
        u = self.loc(name)
        if u is not None:
            u.value = float(v)
        return self

    def vec2(self, name: str, x: float, y: float) -> "Program":
        u = self.loc(name)
        if u is not None:
            u.value = (float(x), float(y))
        return self

    def vec3(self, name: str, x: float, y: float, z: float) -> "Program":
        u = self.loc(name)
        if u is not None:
            u.value = (float(x), float(y), float(z))
        return self

    def vec4(self, name: str, x: float, y: float, z: float, w: float) -> "Program":
        u = self.loc(name)
        if u is not None:
            u.value = (float(x), float(y), float(z), float(w))
        return self

    def floatv(self, name: str, v: np.ndarray) -> "Program":
        u = self.loc(name)
        if u is not None:
            u.write(v)
        return self

    def intv(self, name: str, v: np.ndarray) -> "Program":
        u = self.loc(name)
        if u is not None:
            u.write(v)
        return self

    def vec3v(self, name: str, v: np.ndarray) -> "Program":
        u = self.loc(name)
        if u is not None:
            u.write(v)
        return self

    def mat3(self, name: str, v: np.ndarray) -> "Program":
        u = self.loc(name)
        if u is not None:
            u.write(v)
        return self

    def tex(self, name: str, unit: int, texture: Optional[moderngl.Texture]) -> "Program":
        """Binds ``texture`` to ``unit`` and points the named sampler at it."""
        if texture is not None:
            texture.use(unit)
        return self.int(name, unit)

    def vao(
        self, content: Sequence[Tuple[moderngl.Buffer, str, str]], ctx: Optional[moderngl.Context] = None
    ) -> moderngl.VertexArray:
        """
        A vertex array bound to this program.

        In GL a VAO and a program are independent, so the browser build shared
        one quad VAO across four programs. moderngl ties them together, so the
        same buffer gets several thin VAO objects instead. They are built at init
        and cost one object each; no extra draw calls, no extra uploads.
        """
        return (ctx or self.ctx).vertex_array(self.handle, list(content))


def create_texture(
    ctx: moderngl.Context,
    width: int,
    height: int,
    components: int = 4,
    dtype: str = "f4",
    data: Optional[bytes] = None,
    filter: int = moderngl.NEAREST,
) -> moderngl.Texture:
    tex = ctx.texture((width, height), components, data=data, dtype=dtype)
    tex.filter = (filter, filter)
    tex.repeat_x = False
    tex.repeat_y = False
    return tex


def create_framebuffer(
    ctx: moderngl.Context, attachments: Iterable[moderngl.Texture]
) -> moderngl.Framebuffer:
    fbo = ctx.framebuffer(color_attachments=list(attachments))
    return fbo


class FullscreenTriangle:
    """
    One oversized triangle covering the viewport. Cheaper than a quad and it has
    no diagonal seam where two halves would meet.
    """

    def __init__(self, ctx: moderngl.Context) -> None:
        self.ctx = ctx
        self.vbo = ctx.buffer(np.array([-1, -1, 3, -1, -1, 3], dtype="f4").tobytes())
        self._vaos: Dict[int, moderngl.VertexArray] = {}

    def bind(self, program: Program) -> None:
        """Builds this program's VAO once, at init."""
        self._vaos[id(program)] = program.vao([(self.vbo, "2f", "aPos")], self.ctx)

    def draw(self, program: Program) -> None:
        self._vaos[id(program)].render(moderngl.TRIANGLES, vertices=3)


FULLSCREEN_VS = (
    VERSION
    + """
layout(location = 0) in vec2 aPos;
out vec2 vUv;
void main() {
  vUv = aPos * 0.5 + 0.5;
  gl_Position = vec4(aPos, 0.0, 1.0);
}
"""
)
