"""
Branching lightning, built by recursive midpoint subdivision.

Take a straight segment, displace its midpoint perpendicular by a random amount,
recurse into both halves with the displacement halved, and occasionally spawn a
shorter child bolt off the midpoint. Five levels gives 32 segments per trunk and
the branches make it read as electricity rather than as a jagged line.

Built on the CPU because the algorithm is inherently sequential, but built only
on a fire event and then a few more times during the cast — never per frame. The
vertex array is preallocated as an ``array('f')``, which is Python's Float32Array
in everything but name, so a regeneration allocates nothing and uploads through
a memoryview with no copy: one bufferSubData of a few kilobytes.

Expanded to triangles here rather than drawn as GL_LINES because line width is
clamped to 1 px on virtually every driver that matters.
"""

from __future__ import annotations

import math
from array import array
from typing import Optional

import moderngl

from ..config import FX

FLOATS_PER_VERT = 4  # x, y, edge, t
VERTS_PER_SEG = 6

MASK32 = 0xFFFFFFFF


class BoltField:
    def __init__(self, ctx: moderngl.Context, max_verts: int = FX.bolt_max_verts) -> None:
        self.ctx = ctx
        self.max_verts = max_verts
        self.data = array("f", [0.0]) * (max_verts * FLOATS_PER_VERT)
        self._vert_count = 0
        self._rng = 0x9E3779B9
        self._dirty = False

        self.vbo = ctx.buffer(reserve=max_verts * FLOATS_PER_VERT * 4, dynamic=True)
        self._vao: Optional[moderngl.VertexArray] = None

    def bind(self, program) -> None:  # noqa: ANN001 — a render.gl.Program
        """One VAO, built at init."""
        self._vao = program.vao([(self.vbo, "2f 2f", "aPos", "aEdgeT")], self.ctx)

    def _rand(self) -> float:
        """xorshift32. Deterministic, allocation-free, plenty random for a bolt."""
        x = self._rng
        x ^= (x << 13) & MASK32
        x ^= x >> 17
        x ^= (x << 5) & MASK32
        self._rng = x & MASK32
        return (self._rng & 0xFFFFFF) / 0x1000000

    def seed(self, v: int) -> None:
        self._rng = (int(v) | 1) & MASK32

    def generate(self, cx: float, cy: float, reach: float, trunks: int, width: float) -> None:
        """
        Rebuilds every bolt radiating from a palm.

        :param reach: length of a trunk in world units.
        :param trunks: how many bolts fan out.
        """
        self._vert_count = 0
        spin = self._rand() * math.pi * 2.0
        for i in range(trunks):
            a = spin + (i / trunks) * math.pi * 2.0 + (self._rand() - 0.5) * 0.35
            length = reach * (0.55 + 0.75 * self._rand())
            self._subdivide(
                cx,
                cy,
                cx + math.cos(a) * length,
                cy + math.sin(a) * length,
                5,
                length * 0.22,
                width,
                1.0,
            )
        self._dirty = True

    def _subdivide(
        self,
        ax: float,
        ay: float,
        bx: float,
        by: float,
        depth: int,
        offset: float,
        width: float,
        t: float,
    ) -> None:
        if self._vert_count + VERTS_PER_SEG > self.max_verts:
            return
        if depth <= 0:
            self._emit(ax, ay, bx, by, width, t)
            return

        dx = bx - ax
        dy = by - ay
        length = math.sqrt(dx * dx + dy * dy)
        if length < 1e-5:
            return

        # Perpendicular displacement of the midpoint. This is the whole algorithm.
        px = -dy / length
        py = dx / length
        jitter = (self._rand() - 0.5) * 2.0 * offset
        mx = (ax + bx) * 0.5 + px * jitter
        my = (ay + by) * 0.5 + py * jitter

        self._subdivide(ax, ay, mx, my, depth - 1, offset * 0.55, width, t)
        self._subdivide(mx, my, bx, by, depth - 1, offset * 0.55, width * 0.94, t * 0.97)

        # Branch: a shorter fork carrying on roughly in the parent's direction.
        if depth >= 3 and self._rand() < 0.34:
            ang = math.atan2(dy, dx) + (self._rand() - 0.5) * 1.3
            sub = length * (0.35 + 0.4 * self._rand())
            self._subdivide(
                mx,
                my,
                mx + math.cos(ang) * sub,
                my + math.sin(ang) * sub,
                depth - 2,
                offset * 0.5,
                width * 0.55,
                t * 0.55,
            )

    def _emit(self, ax: float, ay: float, bx: float, by: float, width: float, t: float) -> None:
        """One segment as a quad: two triangles, edge coordinate running -1..1 across."""
        dx = bx - ax
        dy = by - ay
        length = math.sqrt(dx * dx + dy * dy)
        if length < 1e-6:
            return
        nx = (-dy / length) * width
        ny = (dx / length) * width

        # Written out longhand rather than through a local helper: a closure here
        # would be built once per segment, and a regeneration emits hundreds.
        d = self.data
        o = self._vert_count * FLOATS_PER_VERT

        d[o] = ax + nx; d[o + 1] = ay + ny; d[o + 2] = 1.0; d[o + 3] = t
        d[o + 4] = ax - nx; d[o + 5] = ay - ny; d[o + 6] = -1.0; d[o + 7] = t
        d[o + 8] = bx + nx; d[o + 9] = by + ny; d[o + 10] = 1.0; d[o + 11] = t

        o += 12
        d[o] = bx + nx; d[o + 1] = by + ny; d[o + 2] = 1.0; d[o + 3] = t
        d[o + 4] = ax - nx; d[o + 5] = ay - ny; d[o + 6] = -1.0; d[o + 7] = t
        d[o + 8] = bx - nx; d[o + 9] = by - ny; d[o + 10] = -1.0; d[o + 11] = t

        self._vert_count += VERTS_PER_SEG

    def clear(self) -> None:
        self._vert_count = 0

    @property
    def count(self) -> int:
        return self._vert_count

    def draw(self) -> None:
        """One upload per regeneration, not per frame."""
        if self._vert_count == 0 or self._vao is None:
            return
        if self._dirty:
            # A memoryview slice, so the upload reads straight out of the array
            # with no intermediate copy.
            self.vbo.write(memoryview(self.data)[: self._vert_count * FLOATS_PER_VERT])
            self._dirty = False
        self._vao.render(moderngl.TRIANGLES, vertices=self._vert_count)
