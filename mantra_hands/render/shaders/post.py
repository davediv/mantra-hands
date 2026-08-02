"""
Bloom, at quarter resolution, one blur pair, no effect-composer stack.

Bloom is what makes this read as magic rather than as a school project, so it
stays — but it stays cheap. A four-tap box downsample straight to 1/4 res gives
the mip chain for free through bilinear filtering.
"""

from __future__ import annotations

from .common import HASH, SRGB, VERSION

BLOOM_DOWN_FS = (
    VERSION
    + """
precision mediump float;
in vec2 vUv;
uniform sampler2D uScene;
uniform vec2 uTexel;
uniform float uThreshold;
uniform float uKnee;
out vec4 oCol;

void main() {
  vec3 c = texture(uScene, vUv + uTexel * vec2(-1.0, -1.0)).rgb;
  c += texture(uScene, vUv + uTexel * vec2(1.0, -1.0)).rgb;
  c += texture(uScene, vUv + uTexel * vec2(-1.0, 1.0)).rgb;
  c += texture(uScene, vUv + uTexel * vec2(1.0, 1.0)).rgb;
  c *= 0.25;

  // Soft knee, so a surface just under the threshold does not pop the moment
  // it crosses.
  float lum = max(c.r, max(c.g, c.b));
  float soft = clamp(lum - uThreshold + uKnee, 0.0, 2.0 * uKnee);
  soft = soft * soft / (4.0 * uKnee + 1e-4);
  float contrib = max(soft, lum - uThreshold) / max(lum, 1e-4);
  oCol = vec4(c * contrib, 1.0);
}
"""
)

BLOOM_BLUR_FS = (
    VERSION
    + """
precision mediump float;
in vec2 vUv;
uniform sampler2D uSrc;
uniform vec2 uDir;
out vec4 oCol;

// Nine taps folded into five samples using linear filtering between texels.
const float W0 = 0.2270270270;
const float W1 = 0.3162162162;
const float W2 = 0.0702702703;
const float O1 = 1.3846153846;
const float O2 = 3.2307692308;

void main() {
  vec3 c = texture(uSrc, vUv).rgb * W0;
  c += texture(uSrc, vUv + uDir * O1).rgb * W1;
  c += texture(uSrc, vUv - uDir * O1).rgb * W1;
  c += texture(uSrc, vUv + uDir * O2).rgb * W2;
  c += texture(uSrc, vUv - uDir * O2).rgb * W2;
  oCol = vec4(c, 1.0);
}
"""
)

#: The one pass that reaches the screen: scene plus bloom, chromatic aberration
#: pulse, vignette, grain, tone map.
#:
#: The aberration is a radial uv offset applied per channel. It costs two extra
#: scene samples and only while a pulse is actually running.
COMPOSITE_FS = (
    VERSION
    + """
precision mediump float;
in vec2 vUv;
uniform sampler2D uScene;
uniform sampler2D uBloom;
uniform float uBloomAmount;
uniform float uAberration;
uniform float uTime;
uniform float uVignette;
uniform float uFlash;
uniform vec3 uFlashColor;
out vec4 oCol;
"""
    + HASH
    + SRGB
    + """
void main() {
  vec2 uv = vUv;
  vec2 fromCenter = uv - 0.5;
  float rad = length(fromCenter);

  vec3 scene;
  if (uAberration > 0.002) {
    // Scaled by radius so the centre of frame — where the person's face is —
    // stays clean and only the edges smear.
    vec2 off = fromCenter * uAberration * (0.25 + rad * 1.4);
    scene.r = texture(uScene, uv + off).r;
    scene.g = texture(uScene, uv).g;
    scene.b = texture(uScene, uv - off).b;
  } else {
    scene = texture(uScene, uv).rgb;
  }

  vec3 bloom = texture(uBloom, uv).rgb;
  vec3 col = scene + bloom * uBloomAmount;

  col += uFlashColor * uFlash;

  col *= mix(1.0, smoothstep(0.95, 0.25, rad), uVignette);
  col = toneMap(col);

  // A little grain keeps flat gradients from banding on 8-bit panels.
  float g = hash21(uv.x * 1024.0 + floor(uTime * 60.0) * 7.0 + uv.y * 57.0).x;
  col += (g - 0.5) * 0.016;

  oCol = vec4(col, 1.0);
}
"""
)

#: Only used while a clip is being recorded, when the composite goes to an
#: offscreen target at the export resolution and this puts it on the screen
#: letterboxed. The normal path composites straight to the default framebuffer
#: and never runs this.
BLIT_FS = (
    VERSION
    + """
precision mediump float;
in vec2 vUv;
uniform sampler2D uSrc;
out vec4 oCol;
void main() {
  oCol = vec4(texture(uSrc, vUv).rgb, 1.0);
}
"""
)
