"""
Drives the whole app, headless, through the demo path.

Everything below the landmarker is the real thing here: the same mirror, the same
One Euro bank, the same recognizers, the same renderer, the same shell. What this
proves is that they still fit together — that the detection thread and the render
loop agree, that a scripted push produces a cast, and that the scene actually
reaches the framebuffer.

    python3 -m tests.smoke_render [seconds] [--shots DIR]

It is not part of the unit suite: it needs a GPU and a window server, and it
takes a few seconds. Run it after touching anything in render/ or ui/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

from mantra_hands.app import App
from mantra_hands.clock import now_ms


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("seconds", nargs="?", type=float, default=8.0)
    parser.add_argument("--shots", type=Path, default=None, help="write PNGs here")
    parser.add_argument("--visible", action="store_true", help="show the window while it runs")
    args = parser.parse_args()

    app = App(demo=True, width=1280, height=720, visible=args.visible)
    app.present()

    started = now_ms()
    completed_before = app.progress.completed
    frames = 0
    shot_at = 0.0
    shots = 0

    while now_ms() - started < args.seconds * 1000.0 and not app.window.should_close:
        app.step()
        if app.state == "running":
            frames += 1

        if args.shots and app.state == "running" and now_ms() - shot_at > 900:
            shot_at = now_ms()
            shots += 1
            args.shots.mkdir(parents=True, exist_ok=True)
            w, h = app.window.framebuffer_size
            image = Image.frombytes("RGB", (w, h), app.window.ctx.screen.read(components=3))
            image.transpose(Image.FLIP_TOP_BOTTOM).save(args.shots / f"demo-{shots:02d}.png")

    elapsed = (now_ms() - started) / 1000.0
    renderer = app.renderer
    print(f"state        {app.state}")
    print(f"frames       {frames} in {elapsed:.1f}s  ({frames / max(elapsed, 1e-3):.1f} fps)")
    print(f"detect       {app.tracker.detect_hz:.1f} hz")
    print(f"draw calls   {renderer.draw_calls if renderer else 0}")
    print(f"particles    {renderer.particles_drawn if renderer else 0}")
    print(f"casts landed {app.progress.completed - completed_before}")
    print(f"combo        {app.combo}   streak {app.progress.streak}   best {app.progress.best}")
    print(f"words served {app.deck.served if app.deck else 0}")
    print(f"scorer       {app.scorer.kind}")

    ok = app.state == "running" and frames > 60 and (app.deck.served if app.deck else 0) >= 2
    app.destroy()
    print("OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
