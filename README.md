# Mantra Hands

Push your palm at the camera and say the English word. If you said it right, the spell fires.

The element is decided by the word's part of speech, and **that mapping is the lesson**:

| Part of speech | Element   | What it looks like                                  |
| -------------- | --------- | --------------------------------------------------- |
| verb           | fire      | orange-red embers drifting upward                    |
| noun           | ice       | cyan crystalline shards that fall and shatter        |
| adjective      | lightning | yellow-white branching bolts                         |

Built for Gen Z students at an English immersion camp in Kampung Inggris, Pare. It runs entirely
on the machine in front of you, and it works with no internet at all.

---

## Quick start

```bash
pip install -r requirements.txt
python3 -m mantra_hands.vendor    # fetches the 7.5 MB hand model, once
python3 -m mantra_hands
```

`vendor` is the only thing in this project that ever opens a socket, and it is not run
automatically — an app that quietly downloads something on first launch is exactly the failure
this one was written to avoid. Copy the directory to a USB stick, a Raspberry Pi, a laptop with
no network: after that one command it never calls out to anything again.

On macOS the terminal you launch from needs camera permission the first time
(System Settings → Privacy & Security → Camera).

```bash
python3 -m unittest discover -s tests -t .   # gesture recognizers and the mirror transform
python3 -m mantra_hands --help
```

### Demo mode

```bash
python3 -m mantra_hands --demo
```

No camera, no model, scripted hands that charge and cast on a loop. Everything below the
landmarker is the real code path — the same mirror, the same filters, the same recognizers, the
same renderer. Use it to show the app on a projector, or to check the gesture and render path
without standing in front of a webcam.

### Controls

| Key      | Does                                                     |
| -------- | -------------------------------------------------------- |
| `d`      | performance HUD                                           |
| `s`      | settings                                                  |
| `m`      | mirror the camera                                         |
| `p`      | low power mode                                            |
| `[` `]`  | effect intensity, 40–150%                                 |
| `r`      | record a clip — 3-2-1 countdown, 30 s max                 |
| `Esc`    | close the settings panel                                  |
| `q`      | quit                                                      |

```bash
python3 -m mantra_hands --camera 1 --size 1600x900 --fullscreen
```

---

## Swapping in your own word list

**Edit one file: `mantra_hands/content/deck.json`.**

```json
{
  "name": "Batch 14 — speaking test",
  "words": [
    { "word": "borrow", "pos": "verb", "cefr": "A2", "gloss": "take something and give it back later" }
  ]
}
```

- `pos` must be exactly `verb`, `noun` or `adjective`. That is what picks the element.
- `gloss` is the one-line English hint under the word. Keep it short — it is read at a glance.
- `cefr` is for your own reference; the app does not filter on it.

A bad row is skipped with a warning naming the row (`-v` to see them), rather than taking the app
down. Order does not matter: the deck shuffles and then guarantees no two consecutive words share
a part of speech, so a student meets fire, ice and lightning inside their first three casts and
works the rule out without anyone explaining it.

The shipped deck is 120 words at CEFR A2–B1: 40 everyday verbs, 40 TOEFL-frequent nouns, and 40
descriptive adjectives that earn marks in a speaking test.

### Without editing the package

```bash
python3 -m mantra_hands --deck batch14.json
```

The built-in deck stays as the fallback, so a typo in your file cannot leave a class with a blank
screen.

---

## The gesture

**Charge** — open palm facing the camera, held steady.

- All four fingers extended: `distance(tip, wrist) / distance(pip, wrist) > 1.15`. Above 1 the
  fingertip is further out than its own middle joint, which only happens when the finger is
  straight. It is a ratio of two distances, so it is scale-free for free and an index finger and
  a pinky can share one threshold.
- Palm facing the camera: the cross product of `(lm5 − lm0)` and `(lm17 − lm0)`, which is
  perpendicular to the palm, divided by hand size squared and corrected for handedness. A palm
  square to the camera reads about 0.7 and falls off as the cosine of the tilt.
- Held steady: palm drift under 1.4 hand-widths per second.
- Held for at least **400 ms** before a push can register.

**Fire** — a rapid increase in the hand's bounding-box area, which is the only depth proxy worth
trusting. Landmark `z` is not usable for absolute depth and is never read.

The rate is measured over a 110 ms window. A real push from 0.7 m to 0.4 m in 150 ms triples the
area — a rate near 14/s. Landmark jitter over the same window contributes under 0.4/s. The
trigger sits at 2.0/s.

**Both hands at once is a double-power cast.**

### Why a hand entering the frame never misfires

This is the failure the app would be judged on: a hand appearing takes the bounding box from
nothing to everything, which is exactly what a push looks like to an area-growth test. Three
things stop it, and there is a test for each:

1. A push cannot register until a charge has been *held* for 400 ms.
2. The area history is cleared whenever a hand is re-acquired, and the rate reads zero until the
   ring reaches back a full window. For the first ~110 ms of a new hand there is nothing to
   compare against.
3. A charge requires four extended fingers *and* a palm facing the camera *and* stillness, all at
   once.

### Hysteresis, everywhere

Every threshold in `mantra_hands/gestures.py` is a pair — one value to enter a state, a different
one to leave it, plus a minimum hold time. A bare threshold oscillates whenever the signal sits
on the boundary, and a rune strobing on and off reads as a broken app rather than as a hand held
slightly wrong. `tests/test_gestures.py` has a case that wobbles the palm tilt back and forth
across the enter threshold and asserts the charge survives.

---

## Performance architecture

The frame budget is 16.6 ms and it is measured, not assumed. Press `d`.

### Detection is decoupled from rendering

This is the most important decision in the app.

```
camera.read() blocks ──▶ detect_for_video ──▶ mirror ─▶ One Euro ─▶ gestures    ~30 Hz, own thread
                                                            │
                                                            ▼  (extrapolate)
swap() blocks on vsync ─▶ solve ─▶ draw ─▶ bloom ─▶ composite                   display rate, main thread
```

Detection runs on its own clock: a thread whose loop blocks on `VideoCapture.read()`, so it fires
exactly once per decoded camera frame and never twice on the same image. The render loop is the
main thread, paced by vsync. **The render loop never waits on detection.** It reads whatever the
last detection left behind and steps it forward from the last known velocity, with the time step
capped, each channel's step capped in absolute terms, and confidence falling off past one
detection interval. Visuals stay smooth at display rate even though the tracking underneath is at
30.

The recognizers run on the detection thread, at camera rate, on measured landmarks — judging a
push from extrapolated positions would mean judging it partly from our own guess about it. What
they emit, with the power and the both-hands judgement made *at fire time*, crosses to the main
thread through a queue and is applied at the top of the next frame. Effects and scoring touch GL,
and GL belongs to one thread.

Measured on an M1: detection lands at **29.9 Hz** with **13.4 ms** of inference, against the
33 ms the camera gives us.

### No allocation on the frame path

No per-frame temporaries, no list building, no string formatting, no closures created per frame.
Every vector, matrix and buffer is allocated at init:

- Landmarks, velocities and filter state are flat `float32` arrays, one bank for all 42 channels,
  and the One Euro bank updates all of them through preallocated scratch with `out=` on every
  numpy op.
- The per-hand uniform payloads are preallocated arrays, refilled in place each frame.
- Lightning is built into an `array('f')` — Python's `Float32Array` in everything but name — and
  uploaded through a memoryview, so a regeneration copies nothing.
- The 2D chrome patches only the component region that changed, and its quad is scissored to rows
  containing visible pixels instead of shading transparent gaps across a high-DPI display. The
  charge bar, which changes every frame while a hand is charging, is not in that image at all: it
  is two solid quads whose geometry is a uniform.

A GC pause is a dropped frame, and dropped frames are visible in a 60 fps recording.

### The particles

16,384 of them — the spec asks for 12,000 — simulated entirely on the GPU.

- Position and velocity live in a ping-ponged pair of 128×128 `RGBA32F` textures, updated by one
  fragment pass with MRT so both come out of a single draw. There is no CPU-side particle loop
  anywhere.
- Burst directions are precomputed into a static float texture **at init**. Firing a cast
  therefore allocates nothing and uploads nothing: it increments a spawn-id uniform and resets a
  clock. A particle whose stored spawn id no longer matches respawns from the direction texture
  on the GPU.
- They render as **one instanced draw call**.
- All three elements share one shader with a branch on an element uniform — not three materials —
  and all three variants are compiled and drawn once at init.

With two hands up the particles split by column parity, so a double cast is genuinely two bursts
rather than one burst drawn twice.

### Ten draw calls

```
1  particle solver      6  the target word
2  camera frame         7  bloom downsample (quarter res)
3  rune circles         8  bloom blur X
4  lightning bolts      9  bloom blur Y
5  particles           10  composite to the screen
```

A few small scissored draws put the occupied chrome bands on top, and one more pass is used while
a clip is recording. The HUD counts all of them.

Bloom is one pass on a quarter-resolution chain, not an effect-composer stack. It is what makes
this read as magic instead of as a school project, so it stays — it is just kept cheap. Screen
shake modifies the view matrix and nothing else: no re-render, no second pass, nine floats.

### Other things that matter

- **The camera frame goes to the GPU as a texture.** The detection thread converts one BGR frame
  to RGB — the conversion MediaPipe needs anyway — and the render loop uploads that same array.
  Storage is allocated once and reused. Nothing is ever read back inside the frame loop.
- **The video texture uploads once per camera frame**, not once per rendered frame — half of all
  display frames have no new image to upload.
- **Input resolution is 640×480.** Halving the input roughly halves detection cost and is the
  cheapest win available.
- **Every shader is compiled *and drawn* at init**, behind the loading screen, once per element.
  Linking is not enough: plenty of drivers defer the real compile until a draw uses a given
  uniform branch, and paying for that mid-session is a several-hundred-millisecond stall landing
  exactly on the first cast.
- **The resolution is budgeted by pixel count, not by the display's scale factor.** A Retina
  panel would otherwise ask for 3.7 million pixels a frame, and the burst is fill-rate bound long
  before it is anything else bound.
- **The model is vendored into the package**, never fetched at runtime. This runs on camp wifi.

### Low power mode

Half the particles, eighth-resolution bloom, a 30 fps cap and a smaller pixel budget. It turns
itself on when the rolling average frame time is over 20 ms for two seconds — **and says so**,
because a silent quality drop just reads as the app getting worse for no reason. It latches;
climbing back out on its own would mean oscillating for anyone sitting right at the boundary.

The frame cap is `glfwSwapInterval(2)`, so the driver holds the swap. Skipping frames in software
would either spin a core or present a stale buffer.

### Size

| | |
| --- | --- |
| app code | **6,800 lines**, 260 KB |
| tests | 520 lines |
| hand model | 7.5 MB, fetched once, gitignored |

Six dependencies: `mediapipe`, `opencv-python`, `moderngl`, `glfw`, `numpy`, `pillow`. No game
engine, no UI toolkit, no scene graph — the scene is 2D billboards and a solver, so anything
larger would have been overhead with nothing to show for it.

---

## Speech

Behind a `SpeechScorer` interface with two implementations, in `mantra_hands/audio/speech.py`.

**`VoskScorer`** is a small offline model running on the machine, as one persistent session on a
background thread rather than one per charge — opening an audio device takes a couple of hundred
milliseconds and a charge only lasts about that long. Partial results are folded in as they
arrive, so a word said early in the charge is already counted by the time the push lands. The
render loop never polls it. At cast time the app asks `check()` what has already been heard,
which is a synchronous read.

It is opt-in and it never downloads anything: `pip install vosk sounddevice`, put a model where
`MANTRA_HANDS_VOSK_MODEL` points, and pass `--mic`.

**`MockScorer`** returns plausible results after a realistic delay. **The app is fully playable
and demoable offline with the mock** — no microphone, no model, no permission prompt, which is
how it will actually be shown in a room with forty students and one router, and it is what runs
when no model is installed. Its misses are deterministic rather than random (the 4th cast, then
every 7th) so that a three-cast recording always lands while the fizzle is still guaranteed to
show up early enough for everyone to see what a wrong word looks like.

Matching is deliberately generous: one edit per four letters, minimum one, plus a prefix check.
These are learners, many with a first language that does not mark final consonants, being judged
by a recognizer trained mostly on American English. A strict match would punish an accent rather
than a mistake. Failure copy is about the spell, never the speaker.

---

## Layout

```
mantra_hands/
  __main__.py        argv, boot and fatal shell
  app.py             orchestration: the two clocks, cast resolution, framing
  config.py          every tunable in the app
  tracking.py        HandLandmarker, the detection thread, handedness
  landmarks.py       the mirror, One Euro, derived geometry, extrapolation
  gestures.py        the charge/fire hysteresis state machine
  filters.py         One Euro bank, rolling mean, timed ring
  camera.py          cv2.VideoCapture
  perf.py            frame timing and the low-power governor
  demo.py            scripted hands and a synthetic camera for --demo
  clock.py           one monotonic millisecond counter, used everywhere
  fonts.py           the display face, found among what the machine already has
  vendor.py          fetches the hand model, once
  audio/speech.py    SpeechScorer, VoskScorer, MockScorer
  content/           deck.json (edit this) and the loader/sequencer
  render/            OpenGL 3.3 core: renderer, particles, bolts, runes, bloom, shaders
  ui/                shell, perf HUD, recorder, saved progress
  models/            hand_landmarker.task   (vendored, gitignored)
tests/               mirror transform, palm facing, extrapolation, gestures, a headless smoke run
docs/prompt.md       the original spec
```

---

## The mirror

The video is displayed flipped, because that is the only orientation that feels natural to move
in. Every landmark `x` is flipped to `1 − x` the instant it arrives, in `HandFrame.ingest`,
before anything downstream touches it. **That is the only mirror in the codebase.**

Get it wrong and nothing throws — the app just feels subtly, maddeningly reversed, and you will
spend an evening looking in the wrong place. `tests/test_mirror.py` pins it, including that
mirroring happens *before* the geometry is derived and that the palm-facing sign comes out right
for both hands in both facings.

The `m` key is a *display* decision: the landmark pipeline stays mirrored either way, and the
renderer flips the camera frame and the palm-to-world mapping together so they can never
disagree.

**One thing to know:** MediaPipe labels handedness as though the input image were already
mirrored, and the app feeds it the raw frame — so the reported label is the opposite of the
anatomical hand. That convention has quietly changed between MediaPipe releases and it is the
sign of the palm-facing test, so `Tracker._calibrate` watches for the exact failure it would
cause (an open palm reading strongly *negative* for over a second and a half) and flips it. The
first successful charge locks it in. The HUD shows which convention is live.

---

## Recording

Before the countdown the render target switches to a true 1080×1920 and the window letterboxes,
so the framing during the countdown is exactly the framing that gets exported. That resize is one
reallocation, not a render pass. Low power records 720×1280.

The composited frame is read back from that target and encoded by OpenCV. That readback is the
one place this costs something a browser would not have charged — there is no compositor stream
to tap on a desktop — and it only runs while a clip is actually recording. The chrome is not in
the file: the camera and the effects are already composited in the one target being drawn, which
is the whole reason the app renders the video through GL rather than stacking a video widget
under it.

Clips land next to whatever you launched from, as `mantra-hands-001.mp4`.

---

## Known limits

- **Detection is on a thread, not a process.** MediaPipe releases the GIL inside inference, so
  the 13 ms of C++ does not block the render loop — but the Python around it does contend.
  Moving the landmarker into a `multiprocessing` worker with shared memory is the next thing
  worth doing if a machine turns out to be too slow.
- **MediaPipe's GPU delegate is unusable outside Linux.** On macOS it constructs happily and then
  aborts the whole process inside `Image::ConvertToGpu` on the first detection — a `SIGABRT` out
  of C++, with nothing to catch. So the delegate is chosen by platform up front rather than
  caught after it fails: GPU on Linux, CPU everywhere else, `MANTRA_HANDS_DELEGATE=gpu` to
  override. CPU inference at 640×480 leaves plenty of headroom on anything recent; it will be the
  first thing to hurt on an old laptop.
- **The display face is whatever condensed face the machine already has** — DIN Condensed on
  macOS, DejaVu Sans Condensed on Linux, Arial Narrow on Windows. Deliberate: bundling one would
  put a licence and a megabyte into a project whose whole point is that it runs from a USB stick.
- **The microphone needs `--mic`.** The stored preference starts off and the scorer honours it
  even when you start with the microphone. Turn it on once and it persists.
- **Frame rates quoted from a headless run are meaningless.** Verify on the target machine with
  the HUD.

---

## Verification

- `python3 -m unittest discover -s tests -t .` — 31 tests over the mirror transform, palm
  facing, extrapolation clamps, and the full charge/fire state machine, including the "a hand
  entering the frame never misfires" case and the hysteresis wobble. They import nothing from
  `render/`, so they run anywhere Python and numpy do.
- `python3 -m tests.smoke_render 10 --shots ./shots` — drives the whole app headless through the
  demo path: real thread, real queue, real renderer, real shell, into an off-screen framebuffer,
  asserting that casts actually land. Run it after touching anything in `render/` or `ui/`.
- `--demo` — the same thing, on screen, through the real recognizers and renderer with no camera.
- `d` — read the frame budget on the machine you actually care about.
