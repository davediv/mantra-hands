"""
GLSL fragments shared between programs.

Kept as strings, as they were in the browser build. The only change made in the
port is the version line: ``#version 300 es`` becomes ``#version 330 core``.
Desktop GLSL has accepted ES precision statements since 1.30 — they are ignored
rather than rejected — so every shader body below is byte-identical to the one
the WebGL2 build compiles.
"""

from __future__ import annotations

VERSION = "#version 330 core\n"

#: Cheap hashes. Deterministic per particle, so nothing needs storing.
HASH = """
float hash11(float p) {
  p = fract(p * 0.1031);
  p *= p + 33.33;
  p *= p + p;
  return fract(p);
}
vec2 hash21(float p) {
  vec3 p3 = fract(vec3(p) * vec3(0.1031, 0.1030, 0.0973));
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.xx + p3.yz) * p3.zy);
}
vec3 hash31(float p) {
  vec3 p3 = fract(vec3(p) * vec3(0.1031, 0.1030, 0.0973));
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.xxy + p3.yzz) * p3.zyx);
}
"""

#: A divergence-light swirl field built from sines. Not a real curl noise — a
#: real one costs several noise lookups per axis and this is inside a 16k-pixel
#: pass that runs every frame. Visually it does the same job.
CURL = """
vec3 swirl(vec3 p, float s) {
  return vec3(
    sin(p.y * 1.7 + s * 6.0) - cos(p.z * 1.3 - s * 4.0),
    sin(p.z * 1.9 - s * 5.0) - cos(p.x * 1.5 + s * 3.0),
    sin(p.x * 1.3 + s * 7.0) - cos(p.y * 1.7 - s * 6.0)
  ) * 0.5;
}
"""

#: World space is x in [-aspect, aspect], y in [-1, 1], z toward the viewer.
#:
#: The view matrix carries screen shake and nothing else — no re-render, no
#: second pass. The perspective divide is what makes a burst read as coming *at*
#: the camera rather than spreading across it.
VIEW = """
uniform mat3 uView;
uniform float uAspect;
uniform float uCamZ;

vec4 toClip(vec2 world) {
  vec3 v = uView * vec3(world, 1.0);
  return vec4(v.x / uAspect, v.y, 0.0, 1.0);
}

// Returns xy = projected position, z = the scale factor depth gave it.
vec3 project(vec3 p) {
  float d = max(0.3, uCamZ - p.z);
  float k = uCamZ / d;
  return vec3(p.xy * k, k);
}
"""

SRGB = """
vec3 toneMap(vec3 c) {
  // ACES-ish filmic curve. Cheap, and it keeps the hot core of a burst from
  // clipping to a flat white disc.
  c *= 0.85;
  return clamp((c * (2.51 * c + 0.03)) / (c * (2.43 * c + 0.59) + 0.14), 0.0, 1.0);
}
"""
