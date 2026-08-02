"""
The particle solver and its draw.

One fragment pass over a 128x128 RGBA32F pair (position and velocity), MRT so
both come out of one draw. 16,384 particles updated in ~16k fragments — the whole
thing costs less than a single fullscreen pass at 1080p.

Re-seeding is driven entirely by uniforms: a particle whose stored spawn id no
longer matches uSpawnId respawns from the static direction texture. Firing a cast
therefore uploads nothing and allocates nothing — it increments a float and
resets a clock.
"""

from __future__ import annotations

from .common import CURL, HASH, VERSION, VIEW

SIM_FS = (
    VERSION
    + """
precision highp float;

in vec2 vUv;
layout(location = 0) out vec4 oPos;
layout(location = 1) out vec4 oVel;

uniform sampler2D uPos;
uniform sampler2D uVel;
// xyz = unit burst direction, w = the particle's permanent random seed.
// Filled once at init and never touched again.
uniform sampler2D uSpawn;

uniform float uDt;
uniform float uTime;
uniform int uElement;
uniform int uHandCount;
uniform vec3 uPalm[2];
uniform float uSpawnId[2];
uniform int uMode[2];
uniform float uPower[2];
uniform float uCharge[2];
uniform float uBurstLife;
"""
    + HASH
    + CURL
    + """
const int MODE_CHARGE = 0;
const int MODE_BURST = 1;

void main() {
  // With two hands up the particles alternate by column, so a double cast is
  // genuinely two bursts rather than one burst drawn twice.
  int h = (uHandCount > 1) ? (int(gl_FragCoord.x) & 1) : 0;

  vec4 spawn = texture(uSpawn, vUv);
  vec4 P = texture(uPos, vUv);
  vec4 V = texture(uVel, vUv);
  float seed = spawn.w;
  vec3 palm = uPalm[h];

  bool reseed = abs(V.w - uSpawnId[h]) > 0.5;

  if (uMode[h] == MODE_CHARGE) {
    // Spiral in. The tangential term is what curves the path; without it the
    // particles fall straight at the palm and read as rain.
    vec3 toPalm = palm - P.xyz;
    float d = length(toPalm);
    if (reseed || d < 0.05 || P.w <= 0.0) {
      float a = seed * 6.2831853 + uTime * 0.35;
      float r = 0.30 + 0.34 * hash11(seed * 3.7);
      vec3 off = vec3(cos(a) * r, sin(a) * r * 0.92, (hash11(seed * 11.3) - 0.5) * 0.34);
      oPos = vec4(palm + off, 0.35 + 0.65 * hash11(seed * 5.9));
      oVel = vec4(0.0, 0.0, 0.0, uSpawnId[h]);
      return;
    }
    vec3 nd = toPalm / max(1e-4, d);
    vec3 tang = normalize(cross(nd, vec3(0.0, 0.0, 1.0)) + vec3(1e-4));
    float charge = uCharge[h];
    V.xyz += (nd * (1.5 + 3.4 * charge) + tang * (1.7 + 2.6 * charge)) * uDt;
    V.xyz += swirl(P.xyz * 3.0 + uTime * 0.6, seed) * 0.35 * uDt;
    V.xyz *= exp(-2.0 * uDt);
    P.xyz += V.xyz * uDt;
    P.w -= uDt * 0.5;
    oPos = P;
    oVel = vec4(V.xyz, uSpawnId[h]);
    return;
  }

  // --- burst ---
  if (reseed) {
    vec3 dir = spawn.xyz;
    float speed = (2.1 + 3.6 * hash11(seed * 7.1)) * (0.55 + 0.9 * uPower[h]);
    if (uElement == 2) speed *= 1.45;   // lightning arrives first
    if (uElement == 1) speed *= 0.85;   // ice is heavy
    // Spawned across a shell rather than at a single point. Sixteen thousand
    // additive sprites stacked on one pixel is a white disc, not a burst — the
    // spread is what makes the first frame read as shrapnel.
    float shell = 0.04 + 0.34 * hash11(seed * 13.7);
    // Staggered lives so the front edge and the tail do not die together.
    oPos = vec4(palm + dir * shell, 0.78 + 0.22 * hash11(seed * 4.3));
    oVel = vec4(dir * speed, uSpawnId[h]);
    return;
  }

  vec3 acc = vec3(0.0);
  float drag;
  float lifeScale;

  if (uElement == 0) {
    // FIRE — verbs. Embers rise and wander.
    acc.y += 1.45;
    acc += swirl(P.xyz * 2.2 + uTime * 0.7, seed) * 1.6;
    drag = 1.9;
    lifeScale = 1.0;
  } else if (uElement == 1) {
    // ICE — nouns. Crystalline shards fall and tumble.
    acc.y -= 2.5;
    acc += swirl(P.xyz * 1.1, seed) * 0.4;
    drag = 0.5;
    lifeScale = 1.15;
  } else {
    // LIGHTNING — adjectives. Fast, crackling, gone quickly.
    acc += swirl(P.xyz * 5.0 + uTime * 3.2, seed) * 4.5;
    drag = 3.2;
    lifeScale = 0.72;
  }

  V.xyz += acc * uDt;
  V.xyz *= exp(-drag * uDt);
  P.xyz += V.xyz * uDt;
  P.w -= uDt / max(0.05, uBurstLife * lifeScale);

  oPos = vec4(P.xyz, max(0.0, P.w));
  oVel = vec4(V.xyz, V.w);
}
"""
)

#: One instanced quad per particle. Four vertices, one draw call, no CPU loop.
#: The vertex shader reads the solver's output textures directly, so nothing ever
#: comes back to the CPU.
DRAW_VS = (
    VERSION
    + """
precision highp float;

layout(location = 0) in vec2 aCorner;
layout(location = 1) in vec2 aParticleUv;

uniform sampler2D uPos;
uniform sampler2D uVel;
uniform sampler2D uSpawn;
uniform float uSize;
uniform float uTime;
uniform int uElement;
uniform int uMode[2];
uniform int uHandCount;
// Charge motes fade with the rune they belong to. Burst particles do not —
// they own their own lifetime and finish even if the hand leaves the frame.
uniform float uChargeAlpha[2];
"""
    + VIEW
    + HASH
    + """
out vec2 vCorner;
out float vLife;
out float vSeed;
out float vDepth;
out float vFade;
flat out int vMode;

void main() {
  // Must match the solver's column split exactly, or a particle would be
  // simulated as one hand's and drawn as the other's.
  int h = (uHandCount > 1) ? (gl_InstanceID & 1) : 0;
  int mode = uMode[h];
  vMode = mode;
  vFade = uChargeAlpha[h];

  vec4 P = texture(uPos, aParticleUv);
  vec4 V = texture(uVel, aParticleUv);
  float seed = texture(uSpawn, aParticleUv).w;

  vLife = clamp(P.w, 0.0, 1.0);
  vSeed = seed;

  vec3 proj = project(P.xyz);
  vDepth = proj.z;

  // Depth magnification is capped. A particle almost at the lens should be a
  // bright streak going past, not a sprite covering a third of the frame.
  float size = uSize * min(proj.z, 3.6);
  // Charge motes are small and even; burst particles start big and shrink as
  // they burn out.
  size *= (mode == 0) ? (0.55 + 0.5 * seed) : (0.45 + 0.9 * vLife);
  if (uElement == 1 && mode == 1) size *= 1.25;   // ice shards read as chunks

  vec2 corner = aCorner;

  if (uElement == 2 && mode == 1) {
    // Lightning draws as a streak along its own screen-space velocity.
    vec2 dir = V.xy;
    float len = length(dir);
    vec2 axis = len > 1e-3 ? dir / len : vec2(0.0, 1.0);
    corner = vec2(corner.x * 0.22, corner.y * (1.0 + min(2.2, len * 0.55)));
    corner = vec2(corner.x * axis.y + corner.y * axis.x, -corner.x * axis.x + corner.y * axis.y);
  } else if (uElement == 1 && mode == 1) {
    // Ice shards tumble.
    float a = seed * 6.2831853 + uTime * (1.4 + 3.0 * hash11(seed * 2.3));
    float c = cos(a);
    float s = sin(a);
    corner = vec2(corner.x * c - corner.y * s, corner.x * s + corner.y * c);
  }

  vCorner = aCorner;
  gl_Position = toClip(proj.xy + corner * size);
}
"""
)

DRAW_FS = (
    VERSION
    + """
precision mediump float;
// int defaults to highp in a vertex shader and mediump in a fragment shader,
// and a uniform shared across both stages must match in precision or the
// program will not link. uElement is shared, so this is not optional.
precision highp int;

in vec2 vCorner;
in float vLife;
in float vSeed;
in float vDepth;
in float vFade;
flat in int vMode;

uniform int uElement;
uniform vec3 uColor;
uniform vec3 uHot;
uniform float uIntensity;

out vec4 oCol;

void main() {
  float r = length(vCorner);
  float a;

  if (uElement == 0) {
    // Soft round ember with a hot core.
    a = exp(-3.4 * r * r);
  } else if (uElement == 1) {
    // Faceted shard: a diamond with one edge catching the light.
    float dm = abs(vCorner.x) * 1.45 + abs(vCorner.y) * 0.8;
    a = smoothstep(1.0, 0.5, dm);
    a *= 0.55 + 0.45 * step(0.0, vCorner.x * vCorner.y);
  } else {
    a = exp(-5.5 * vCorner.x * vCorner.x) * exp(-1.1 * vCorner.y * vCorner.y);
  }

  if (vMode == 0) a = exp(-3.0 * r * r) * vFade;

  // Squared falloff so the burst empties out fast enough to read as an impact
  // rather than as a fog bank.
  a *= vLife * vLife;
  if (a < 0.004) discard;

  // Only partway to the hot colour even at full life. Taking it all the way
  // would make every fresh particle near-white, and the element would only
  // become recognisable as it died.
  float heat = clamp(vLife * 1.35 + 0.15 * vSeed, 0.0, 1.0);
  vec3 col = mix(uColor, uHot, heat * heat * 0.7);
  // Depth brightening: particles nearest the camera are the ones that read.
  col *= 0.6 + 0.45 * clamp(vDepth, 0.0, 3.0);

  // Sixteen thousand additive sprites saturate to white long before any one of
  // them is bright. Per-particle output is scaled down so the *density* is what
  // makes the burst hot — which is also what leaves the bloom something to do.
  oCol = vec4(col * a * uIntensity * 0.42, a * 0.42);
}
"""
)
