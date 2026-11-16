"""Demo: multi-frame animation compositing + palette-consistent GIF export.

Builds three small self-made animation groups, exports each as one animated
GIF via gb_pipeline.export_animation, and reports for every group: frame
count, total duration, per-frame pixel correctness (GIF re-decoded vs the
composited frames), and a size comparison against direct per-frame PNG
export. Also demonstrates the mismatched-frame-size error and that the
single-frame export stays pixel-identical to the source.

Run:  python scripts/animation_demo.py
Outputs land in artifacts/.
"""

import io
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gb_pipeline as gp

ARTIFACTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "artifacts")

# GB-flavoured color sets (RGB555-snapped).
DMG = [tuple(c) for c in gp.DMG_RAMP.tolist()]                      # 4-shade mono
COLOUR = [(224, 248, 208), (136, 192, 112), (190, 80, 72), (52, 104, 86),
          (8, 24, 32), (238, 196, 92)]                            # 6-colour scene


def _base_image(w, h, colors):
    """Deterministic striped background cycling `colors` per 8px column."""
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    for x in range(w):
        arr[:, x] = colors[(x // 8) % len(colors)]
    return Image.fromarray(arr, "RGB")


def _delta(w, h, rects, color):
    """An RGBA delta frame: `color` painted in each (x, y, rw, rh) rect."""
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    for x, y, rw, rh in rects:
        arr[y:y + rh, x:x + rw] = color + (255,)
    return Image.fromarray(arr, "RGBA")


def build_banner_colour():
    """128x96, 5 frames: a marker sweeping a striped colour banner.

    Frame 0 is the base scene; frames 1-4 are RGBA deltas whose drawn pixels
    are cleared back to the background after display (dispose=background), so
    the marker appears to move over the static banner.
    """
    w, h = 128, 96
    base = _base_image(w, h, COLOUR)
    frames = [base]
    for i in range(4):
        frames.append(_delta(w, h, [(16 + i * 24, 40, 16, 16)], COLOUR[5]))
    return {
        "name": "banner_colour",
        "frames": frames,
        "delays": [150, 150, 100, 100, 150],
        "disposes": ["keep", "background", "background", "background", "background"],
        "loop": 0,
        "background": COLOUR[0],
        "first_frame_base": True,
        "palette": None,
    }


def build_blink_mono():
    """160x144, 4 frames: a mono face blinking (open/closed/open/closed).

    The closed-eye deltas use dispose=previous, so each is rolled back to the
    open-eyed base before the next frame. Palette locked to the DMG ramp.
    """
    w, h = 160, 144
    arr = np.full((h, w, 3), DMG[0], dtype=np.uint8)          # lightest
    arr[32:112, 40:120] = DMG[1]                              # head
    for ex in (64, 96):                                       # open eyes
        arr[56:68, ex:ex + 8] = DMG[3]
    arr[88:96, 68:92] = DMG[2]                                # mouth
    base = Image.fromarray(arr, "RGB")
    closed = _delta(w, h, [(64, 60, 8, 4), (96, 60, 8, 4)], DMG[2])
    empty = _delta(w, h, [], DMG[2])
    return {
        "name": "blink_mono",
        "frames": [base, closed, empty, closed],
        "delays": [120, 120, 120, 120],
        "disposes": ["keep", "previous", "keep", "previous"],
        "loop": 0,
        "background": DMG[0],
        "first_frame_base": True,
        "palette": np.array(DMG, dtype=np.uint8),
    }


def build_logo_badge():
    """160x144, 5 frames: a badge wiping in column by column (dispose=keep).

    Deltas accumulate on the base, so the badge builds up over the sequence.
    Loops 3 times instead of forever (loop count is configurable).
    """
    w, h = 160, 144
    base = _base_image(w, h, [DMG[3], DMG[3]])                # solid darkest
    frames = [base]
    for i in range(4):
        frames.append(_delta(w, h, [(48 + i * 16, 48, 16, 48)], DMG[i % 2 + 1]))
    return {
        "name": "logo_badge",
        "frames": frames,
        "delays": [200, 200, 200, 200, 200],
        "disposes": "keep",
        "loop": 3,
        "background": DMG[3],
        "first_frame_base": True,
        "palette": np.array(DMG, dtype=np.uint8),
    }


def gif_frames_rgb(data):
    """Decode a GIF's frames to a list of RGB arrays."""
    im = Image.open(io.BytesIO(data))
    out = []
    for i in range(im.n_frames):
        im.seek(i)
        out.append(np.asarray(im.convert("RGB"), dtype=np.uint8))
    return out


def png_bytes(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return len(buf.getvalue())


def run_group(spec):
    data, info = gp.export_animation(
        spec["frames"],
        delays=spec["delays"],
        disposes=spec["disposes"],
        loop=spec["loop"],
        background=spec["background"],
        first_frame_base=spec["first_frame_base"],
        palette=spec["palette"],
    )
    # Per-frame pixel correctness: re-composite independently and compare
    # every decoded GIF frame against the composited canvas, pixel for pixel.
    composited, _ = gp.compose_animation(
        spec["frames"], delays=spec["delays"], disposes=spec["disposes"],
        background=spec["background"],
        first_frame_base=spec["first_frame_base"],
    )
    bg = spec["background"]
    want = [np.asarray(gp._flatten_rgba(c, bg), dtype=np.uint8) for c in composited]
    got = gif_frames_rgb(data)
    per_frame_ok = [np.array_equal(g, w) for g, w in zip(got, want)]
    assert len(got) == len(want) == info["n_frames"]

    # Size comparison vs direct per-frame export (one PNG per frame, which is
    # what the app's folder export produces today).
    direct_bytes = sum(
        png_bytes(Image.fromarray(w, "RGB")) for w in want
    )
    out_path = os.path.join(ARTIFACTS, f"anim_{spec['name']}.gif")
    with open(out_path, "wb") as fh:
        fh.write(data)

    print(f"  {spec['name']}  ({info['size'][0]}x{info['size'][1]})")
    print(f"    frames: {info['n_frames']}   delays: {info['delays_ms']} ms"
          f"   total: {info['total_duration_ms']} ms"
          f"   loop: {'forever' if info['loop'] == 0 else info['loop']}")
    print(f"    per-frame pixel check: {sum(per_frame_ok)}/{len(per_frame_ok)} exact"
          f"   palette colors: {info['palette_colors']}")
    print(f"    GIF: {info['size_bytes']} B   direct per-frame PNGs: {direct_bytes} B"
          f"   ratio: {info['size_bytes'] / direct_bytes:.2f}x"
          f"   -> {os.path.relpath(out_path)}")
    assert all(per_frame_ok), f"{spec['name']}: pixel mismatch"
    return info


def main():
    os.makedirs(ARTIFACTS, exist_ok=True)
    print("Animation export demo -- three groups")
    groups = [build_banner_colour(), build_blink_mono(), build_logo_badge()]
    infos = [run_group(spec) for spec in groups]

    print()
    print("Frame counts / total durations:")
    for spec, info in zip(groups, infos):
        print(f"  {spec['name']}: {info['n_frames']} frames,"
              f" {info['total_duration_ms']} ms")

    print()
    print("Single-frame export pixel-identity:")
    src = groups[0]["frames"][0]
    data, info = gp.export_animation([src], delays=[100], background=COLOUR[0])
    got = gif_frames_rgb(data)[0]
    same = np.array_equal(got, np.asarray(src, dtype=np.uint8))
    print(f"  single frame -> GIF -> decode: {'IDENTICAL' if same else 'DIFFERS'}"
          f" ({info['n_frames']} frame, {info['size_bytes']} B)")
    assert same

    print()
    print("Mismatched frame sizes:")
    try:
        gp.export_animation([Image.new("RGB", (16, 16)), Image.new("RGB", (24, 16))])
    except ValueError as exc:
        print(f"  ValueError raised as expected: {exc}")
    else:
        raise AssertionError("expected a ValueError for mismatched frame sizes")

    print()
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
