"""The camera frame, the rune, the bolts and the word."""

from __future__ import annotations

from .common import HASH, VERSION, VIEW

#: The camera frame.
#:
#: Overscanned slightly so screen shake — which is a view matrix change and
#: nothing else — can never pull a black edge into frame. The video texture is
#: sampled with the u axis flipped, matching the mirror the user sees.
VIDEO_VS = (
    VERSION
    + """
precision highp float;
layout(location = 0) in vec2 aCorner;
"""
    + VIEW
    + """
uniform vec2 uCoverScale;
uniform float uOverscan;
// +1 mirrored, which is the default and the only orientation that feels
// natural to move in. -1 shows the raw camera frame.
uniform float uMirror;

out vec2 vUv;

void main() {
  // Cover fit: whichever axis is short gets its uv range squeezed, so the frame
  // fills the window without letterboxing and without stretching faces.
  //   x is negated for the mirror,
  //   y is negated because clip y points up and texture v points down.
  // Renderer.world_x_from_video is the exact inverse of this and has to stay so.
  vUv.x = 0.5 - aCorner.x * 0.5 * uCoverScale.x * uMirror;
  vUv.y = 0.5 - aCorner.y * 0.5 * uCoverScale.y;
  gl_Position = toClip(aCorner * vec2(uAspect, 1.0) * uOverscan);
}
"""
)

VIDEO_FS = (
    VERSION
    + """
precision mediump float;
in vec2 vUv;
uniform sampler2D uVideo;
uniform float uDim;
out vec4 oCol;

void main() {
  vec3 c = texture(uVideo, vUv).rgb;
  // A touch of lift toward the base navy keeps the effects sitting *on* the
  // person rather than fighting the camera's own contrast.
  c = mix(c, c * vec3(0.82, 0.88, 1.06) + vec3(0.02, 0.03, 0.06), 0.55);
  oCol = vec4(c * uDim, 1.0);
}
"""
)

#: The rune circle, drawn procedurally — no texture, one instanced quad per hand,
#: distance fields all the way down. It rotates, scales and brightens with hold
#: time, and on a failed cast it cracks and fizzles magenta.
RUNE_VS = (
    VERSION
    + """
precision highp float;
layout(location = 0) in vec2 aCorner;
"""
    + VIEW
    + """
uniform vec2 uRunePos[2];
uniform float uRuneSize[2];

out vec2 vP;
flat out int vId;

void main() {
  int i = gl_InstanceID;
  vP = aCorner;
  vId = i;
  gl_Position = toClip(uRunePos[i] + aCorner * uRuneSize[i]);
}
"""
)

RUNE_FS = (
    VERSION
    + """
precision mediump float;

in vec2 vP;
flat in int vId;

uniform float uTime;
uniform float uCharge[2];
uniform float uAlpha[2];
uniform float uFail[2];
uniform vec3 uColor;
uniform vec3 uHot;
uniform vec3 uFailColor;
uniform float uIntensity;

out vec4 oCol;
"""
    + HASH
    + """
float ring(float r, float radius, float w) {
  return smoothstep(w, 0.0, abs(r - radius));
}

void main() {
  float charge = uCharge[vId];
  float alpha = uAlpha[vId];
  float fail = uFail[vId];
  if (alpha < 0.004) discard;

  vec2 p = vP;
  float r = length(p);
  if (r > 1.02) discard;

  float a = atan(p.y, p.x);
  float t = uTime * (0.45 + 1.6 * charge);

  // On a failed cast the circle breaks apart along a few radial faults and the
  // pieces drift. The joke is at the spell's expense, never the speaker's.
  if (fail > 0.001) {
    float shard = floor((a / 6.2831853 + 0.5) * 5.0);
    float push = fail * (0.05 + 0.09 * hash11(shard * 3.1));
    r += push;
    a += fail * (hash11(shard * 7.7) - 0.5) * 0.5;
  }

  float m = 0.0;
  // Outer boundary.
  m += ring(r, 0.95, 0.014) * 1.1;
  // Tick ring: 24 marks, slowly turning.
  float ticks = step(0.45, fract((a / 6.2831853) * 24.0 + t * 0.08));
  m += ring(r, 0.845, 0.026) * (0.25 + 0.75 * ticks);
  // Counter-rotating glyph band.
  float glyph = step(0.58, fract((a + t) / 6.2831853 * 11.0));
  m += ring(r, 0.70, 0.040) * glyph * 0.95;
  // Spokes reaching in toward the palm.
  float spoke = step(0.88, fract((a - t * 0.7) / 6.2831853 * 6.0));
  m += smoothstep(0.62, 0.18, r) * spoke * 0.55;
  // Inner ring pulses with the charge.
  m += ring(r, 0.34, 0.016) * (0.55 + 0.45 * sin(t * 4.0));

  // Cracks, only when failing.
  if (fail > 0.001) {
    float crack = step(0.965, fract(a / 6.2831853 * 9.0 + sin(r * 22.0) * 0.12));
    m += crack * smoothstep(1.0, 0.2, r) * fail * 1.4;
  }

  // Core glow, sitting right on the palm.
  float core = exp(-7.0 * r * r) * (0.25 + 0.75 * charge);

  vec3 col = mix(uColor, uHot, charge * charge);
  col = mix(col, uFailColor, fail);

  float lum = (m * (0.35 + 0.9 * charge) + core) * alpha;
  oCol = vec4(col * lum * uIntensity, lum);
}
"""
)

#: Lightning bolts. Geometry is built on the CPU by recursive subdivision (see
#: bolts.py) into a preallocated buffer and expanded to triangles there, because
#: GL line width is capped at 1 px almost everywhere.
BOLT_VS = (
    VERSION
    + """
precision highp float;
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec2 aEdgeT;
"""
    + VIEW
    + """
out float vEdge;
out float vT;

void main() {
  vEdge = aEdgeT.x;
  vT = aEdgeT.y;
  gl_Position = toClip(aPos);
}
"""
)

BOLT_FS = (
    VERSION
    + """
precision mediump float;
in float vEdge;
in float vT;
uniform vec3 uColor;
uniform vec3 uHot;
uniform float uFade;
uniform float uIntensity;
out vec4 oCol;

void main() {
  // Hard white core, soft coloured falloff — the shape of every good bolt.
  float d = abs(vEdge);
  float core = smoothstep(0.35, 0.0, d);
  float halo = smoothstep(1.0, 0.15, d);
  float a = (core * 1.6 + halo * 0.5) * uFade * vT;
  if (a < 0.004) discard;
  vec3 col = mix(uColor, uHot, core);
  oCol = vec4(col * a * uIntensity, a);
}
"""
)

#: The target word, huge across the burst, then dissolving. Rasterised to a
#: texture once when the word changes — never per frame — so this pass is a
#: single quad and one sampler.
WORD_VS = (
    VERSION
    + """
precision highp float;
layout(location = 0) in vec2 aCorner;
"""
    + VIEW
    + """
uniform vec2 uWordCenter;
uniform vec2 uWordSize;

out vec2 vUv;

void main() {
  vUv = aCorner * 0.5 + 0.5;
  vUv.y = 1.0 - vUv.y;
  gl_Position = toClip(uWordCenter + aCorner * uWordSize);
}
"""
)

WORD_FS = (
    VERSION
    + """
precision mediump float;
in vec2 vUv;
uniform sampler2D uWord;
uniform float uDissolve;
uniform float uAlpha;
uniform vec3 uColor;
uniform vec3 uHot;
uniform float uIntensity;
out vec4 oCol;
"""
    + HASH
    + """
void main() {
  if (uAlpha < 0.004) discard;
  float mask = texture(uWord, vUv).a;
  if (mask < 0.01) discard;

  // Dissolve from a hash so the word breaks into flakes rather than fading
  // like an opacity transition. Two octaves: the coarse one tears the word
  // into regions, the fine one eats the edges of those regions, which is what
  // keeps it from reading as JPEG blocks.
  vec2 coarse = floor(vUv * vec2(46.0, 13.0));
  vec2 fine = floor(vUv * vec2(320.0, 90.0));
  float n = hash21(coarse.x * 57.0 + coarse.y * 13.0).x * 0.68
          + hash21(fine.x * 31.0 + fine.y * 71.0).x * 0.32;
  float edge = uDissolve * 1.35;
  float keep = smoothstep(edge - 0.22, edge, n);
  if (keep < 0.01) discard;

  // The rim of the dissolving edge burns hot.
  float rim = smoothstep(0.0, 0.28, n - edge + 0.22) * (1.0 - smoothstep(0.28, 0.5, n - edge + 0.22));
  vec3 col = mix(uColor, uHot, 0.35 + 0.65 * rim);
  float a = mask * keep * uAlpha;
  oCol = vec4(col * a * uIntensity * (1.0 + rim * 2.0), a);
}
"""
)

#: The 2D chrome — everything the browser build put in the DOM — as one
#: premultiplied RGBA texture the size of the window, plus a solid-colour quad
#: for the charge bar. See ui/shell.py for why those are two different things.
OVERLAY_VS = (
    VERSION
    + """
precision highp float;
layout(location = 0) in vec2 aCorner;
out vec2 vUv;
void main() {
  vUv = vec2(aCorner.x * 0.5 + 0.5, 0.5 - aCorner.y * 0.5);
  gl_Position = vec4(aCorner, 0.0, 1.0);
}
"""
)

OVERLAY_FS = (
    VERSION
    + """
precision mediump float;
in vec2 vUv;
uniform sampler2D uLayer;
out vec4 oCol;
void main() {
  vec4 c = texture(uLayer, vUv);
  if (c.a < 0.004) discard;
  oCol = c;
}
"""
)

#: A texture in an arbitrary rectangle of the screen — the perf HUD's panel.
PANEL_VS = (
    VERSION
    + """
precision highp float;
layout(location = 0) in vec2 aCorner;
uniform vec4 uRect;   // x, y, halfWidth, halfHeight in NDC
out vec2 vUv;
void main() {
  vUv = vec2(aCorner.x * 0.5 + 0.5, 0.5 - aCorner.y * 0.5);
  gl_Position = vec4(uRect.xy + aCorner * uRect.zw, 0.0, 1.0);
}
"""
)

PANEL_FS = (
    VERSION
    + """
precision mediump float;
in vec2 vUv;
uniform sampler2D uLayer;
out vec4 oCol;
void main() {
  vec4 c = texture(uLayer, vUv);
  if (c.a < 0.004) discard;
  oCol = c;
}
"""
)

#: A solid rectangle in normalized device coordinates. The charge bar is drawn
#: with this rather than rasterised into the overlay, because it changes every
#: frame while a hand is charging and re-rasterising a full-window image at 60 Hz
#: would cost more than the entire scene does.
RECT_VS = (
    VERSION
    + """
precision highp float;
layout(location = 0) in vec2 aCorner;
uniform vec4 uRect;   // x, y, halfWidth, halfHeight in NDC
out vec2 vLocal;
void main() {
  vLocal = aCorner;
  gl_Position = vec4(uRect.xy + aCorner * uRect.zw, 0.0, 1.0);
}
"""
)

RECT_FS = (
    VERSION
    + """
precision mediump float;
in vec2 vLocal;
uniform vec4 uColor;
out vec4 oCol;
void main() {
  oCol = vec4(uColor.rgb * uColor.a, uColor.a);
}
"""
)
