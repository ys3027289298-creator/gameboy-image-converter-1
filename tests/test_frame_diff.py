"""Tests for the animation frame-difference (delta-block) codec in gb_pipeline.

All fixtures are tiny synthetic frames built in-memory -- no image files, no
network, stdlib+numpy only. The codec under test:

* encode_frame_diff / decode_frame_diff -- baseline + per-frame block deltas,
  chained reconstruction on the running (previous reconstructed) frame;
* serialize_frame_diff / deserialize_frame_diff -- self-describing bytes;
* MissingBaselineError -- raised when frame 0 is absent;
* frame_diff_payload_size / frame_diff_full_size -- the volume comparison.
"""

import numpy as np
import pytest

from gb_pipeline import (
    FrameDiff,
    FrameDiffSequence,
    MissingBaselineError,
    decode_frame_diff,
    deserialize_frame_diff,
    encode_frame_diff,
    frame_diff_changed_blocks,
    frame_diff_full_size,
    frame_diff_payload_size,
    frame_diff_pixel_mismatch,
    serialize_frame_diff,
)


def _solid(h, w, color):
    """A solid (h, w, 3) uint8 frame."""
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:, :] = np.asarray(color, dtype=np.uint8)
    return arr


# --------------------------------------------------------------------------
# Lossless round-trip and block selection
# --------------------------------------------------------------------------

def test_lossless_roundtrip_reproduces_every_input_frame():
    f0 = _solid(16, 16, (10, 20, 30))
    f1 = f0.copy()
    f1[0:4, 0:4] = (200, 100, 50)          # block (0, 0) changes
    f2 = f1.copy()
    f2[12:16, 12:16] = (1, 2, 3)           # block (3, 3) changes; block (0,0) stays
    frames = [f0, f1, f2]

    seq = encode_frame_diff(frames, block_size=4, change_threshold=0)
    rebuilt = decode_frame_diff(seq)

    assert len(rebuilt) == 3
    for original, got in zip(frames, rebuilt):
        assert np.array_equal(got, original)
    assert frame_diff_pixel_mismatch(frames, rebuilt) == [0, 0, 0]


def test_only_changed_block_indices_are_recorded():
    f0 = _solid(16, 8, (0, 0, 0))
    f1 = f0.copy()
    f1[0:4, 0:4] = (9, 9, 9)                # row-major block index 0 only
    f2 = f1.copy()
    f2[8:12, 4:8] = (7, 7, 7)              # grid row 2, col 1 -> index 2*2+1 = 5

    seq = encode_frame_diff([f0, f1, f2], block_size=4)

    assert seq.deltas[0].changed.tolist() == [0]
    assert seq.deltas[1].changed.tolist() == [5]
    # Indices are strictly ascending, as the wire format requires.
    for delta in seq.deltas:
        assert np.all(np.diff(delta.changed) > 0)


def test_identical_frame_produces_empty_delta():
    f0 = _solid(8, 8, (4, 5, 6))
    seq = encode_frame_diff([f0, f0.copy(), f0.copy()], block_size=8)

    assert frame_diff_changed_blocks(seq) == [0, 0]
    for delta in seq.deltas:
        assert delta.changed.shape == (0,)
        assert delta.blocks.shape == (0, 8, 8, 3)
    rebuilt = decode_frame_diff(seq)
    assert all(np.array_equal(f, f0) for f in rebuilt)


def test_chain_reconstruction_keeps_earlier_changes_in_later_frames():
    # Block A changes in frame 1; frame 2 changes only block B. The decoder
    # chains on the previous reconstructed frame, so A's change must survive.
    f0 = _solid(16, 16, (0, 0, 0))
    f1 = f0.copy()
    f1[0:8, 0:8] = (10, 10, 10)            # block index 0
    f2 = f1.copy()
    f2[8:16, 8:16] = (20, 20, 20)          # block (1,1) in a 2x2 grid -> index 3

    seq = encode_frame_diff([f0, f1, f2], block_size=8)

    assert seq.deltas[0].changed.tolist() == [0]
    assert seq.deltas[1].changed.tolist() == [3]
    rebuilt = decode_frame_diff(seq)
    assert np.array_equal(rebuilt[2], f2)
    assert np.array_equal(rebuilt[2][0:8, 0:8], f1[0:8, 0:8])


def test_block_size_is_configurable_across_grid_shapes():
    f0 = _solid(16, 16, (0, 0, 0))
    f1 = f0.copy()
    f1[0, 0] = (1, 1, 1)                   # single pixel lives in block index 0

    for block_size, expected_blocks in [(4, 16), (8, 4), (16, 1)]:
        seq = encode_frame_diff([f0, f1], block_size=block_size)
        assert seq.block_size == block_size
        assert seq.deltas[0].changed.tolist() == [0]
        assert seq.deltas[0].blocks.shape == (1, block_size, block_size, 3)
        assert frame_diff_payload_size(seq) == (
            16 * 16 * 3 + block_size * block_size * 3
        )
        assert np.array_equal(decode_frame_diff(seq)[1], f1)
        assert expected_blocks  # marker that each grid size was exercised


# --------------------------------------------------------------------------
# Change threshold
# --------------------------------------------------------------------------

def test_threshold_zero_records_any_differing_pixel():
    f0 = _solid(8, 8, (0, 0, 0))
    f1 = f0.copy()
    f1[3, 3] = (255, 255, 255)             # exactly one pixel differs

    seq = encode_frame_diff([f0, f1], block_size=8, change_threshold=0)
    assert frame_diff_changed_blocks(seq) == [1]
    assert frame_diff_pixel_mismatch([f1], decode_frame_diff(seq)[1:]) == [0]


def test_threshold_skips_blocks_with_few_differing_pixels():
    f0 = _solid(8, 8, (0, 0, 0))
    f1 = f0.copy()
    f1[0, 0] = (255, 0, 0)                 # one differing pixel

    # Threshold 1 means a block needs MORE than one differing pixel, so the
    # block is not recorded and the decoder keeps the previous content.
    seq = encode_frame_diff([f0, f1], block_size=8, change_threshold=1)
    assert frame_diff_changed_blocks(seq) == [0]
    rebuilt = decode_frame_diff(seq)
    assert frame_diff_pixel_mismatch([f1], rebuilt[1:]) == [1]
    assert np.array_equal(rebuilt[1], f0)


def test_threshold_records_block_once_change_count_exceeds_it():
    f0 = _solid(8, 8, (0, 0, 0))
    f1 = f0.copy()
    f1[0, 0] = (255, 0, 0)
    f1[1, 1] = (0, 255, 0)                # two differing pixels > threshold 1

    seq = encode_frame_diff([f0, f1], block_size=8, change_threshold=1)
    assert frame_diff_changed_blocks(seq) == [1]
    assert frame_diff_pixel_mismatch([f1], decode_frame_diff(seq)[1:]) == [0]


def test_threshold_counts_pixels_not_channels():
    # One pixel differs in all three channels: that is still ONE changed pixel.
    f0 = _solid(8, 8, (10, 20, 30))
    f1 = f0.copy()
    f1[0, 0] = (40, 50, 60)

    seq = encode_frame_diff([f0, f1], block_size=8, change_threshold=1)
    assert frame_diff_changed_blocks(seq) == [0]


# --------------------------------------------------------------------------
# Serialization
# --------------------------------------------------------------------------

def test_serialize_roundtrip_is_byte_stable_and_lossless():
    f0 = _solid(16, 16, (1, 2, 3))
    f1 = f0.copy()
    f1[0:8, 0:8] = (9, 8, 7)
    frames = [f0, f1]

    seq = encode_frame_diff(frames, block_size=8, change_threshold=0)
    blob = serialize_frame_diff(seq)
    assert isinstance(blob, bytes) and len(blob) > 0
    # Deterministic: encoding the same frames twice yields identical bytes.
    assert blob == serialize_frame_diff(
        encode_frame_diff(frames, block_size=8, change_threshold=0)
    )

    restored = deserialize_frame_diff(blob)
    assert restored.block_size == 8
    assert restored.change_threshold == 0
    assert (restored.height, restored.width) == (16, 16)
    for a, b in zip(decode_frame_diff(restored), frames):
        assert np.array_equal(a, b)


def test_serialized_blob_includes_index_and_header_overhead():
    f0 = _solid(8, 8, (0, 0, 0))
    f1 = f0.copy()
    f1[0, 0] = (1, 1, 1)
    seq = encode_frame_diff([f0, f1], block_size=8)

    payload = frame_diff_payload_size(seq)          # pure image bytes
    blob = serialize_frame_diff(seq)
    # 30-byte fixed header + per-frame count (4) + one block index (4),
    # on top of the image payload.
    assert len(blob) == payload + 30 + 4 + 4
    assert frame_diff_full_size(seq) == 2 * 8 * 8 * 3


def test_deserialize_rejects_bad_magic_and_version():
    good = serialize_frame_diff(
        encode_frame_diff([_solid(8, 8, (0, 0, 0))], block_size=8)
    )
    bad_magic = b"XXXX" + good[4:]
    with pytest.raises(ValueError, match="magic"):
        deserialize_frame_diff(bad_magic)

    bad_version = good[:4] + bytes([99]) + good[5:]
    with pytest.raises(ValueError, match="version"):
        deserialize_frame_diff(bad_version)

    with pytest.raises(ValueError):
        deserialize_frame_diff(b"short")


def test_deserialize_rejects_trailing_bytes():
    blob = serialize_frame_diff(
        encode_frame_diff([_solid(8, 8, (0, 0, 0)), _solid(8, 8, (1, 1, 1))],
                          block_size=8)
    )
    with pytest.raises(ValueError, match="trailing"):
        deserialize_frame_diff(blob + b"\x00")


# --------------------------------------------------------------------------
# Missing baseline
# --------------------------------------------------------------------------

def test_decode_without_baseline_raises_missingbaselineerror():
    f0 = _solid(8, 8, (0, 0, 0))
    f1 = f0.copy()
    f1[0, 0] = (5, 5, 5)
    seq = encode_frame_diff([f0, f1], block_size=8)
    lost = FrameDiffSequence(
        baseline=None,
        deltas=seq.deltas,
        block_size=seq.block_size,
        change_threshold=seq.change_threshold,
        width=seq.width,
        height=seq.height,
    )

    with pytest.raises(MissingBaselineError, match="baseline"):
        decode_frame_diff(lost)
    # The failure mode is a key-style lookup failure, not an AssertionError.
    assert issubclass(MissingBaselineError, KeyError)


def test_serialized_baseline_loss_raises_but_keeps_surviving_deltas():
    f0 = _solid(8, 8, (0, 0, 0))
    frames = [f0]
    for color in ((1, 1, 1), (2, 2, 2), (3, 3, 3)):
        nxt = frames[-1].copy()
        nxt[0, 0] = color
        frames.append(nxt)
    seq = encode_frame_diff(frames, block_size=8)
    lost = FrameDiffSequence(
        baseline=None,
        deltas=seq.deltas,
        block_size=seq.block_size,
        change_threshold=seq.change_threshold,
        width=seq.width,
        height=seq.height,
    )
    blob = serialize_frame_diff(lost)

    with pytest.raises(MissingBaselineError, match="baseline") as exc_info:
        deserialize_frame_diff(blob)
    # The deltas themselves parse fine and remain available for inspection.
    surviving = exc_info.value.sequence
    assert len(surviving.deltas) == 3
    assert surviving.baseline is None
    assert surviving.width == 8 and surviving.height == 8


# --------------------------------------------------------------------------
# Volume accounting
# --------------------------------------------------------------------------

def test_full_storage_size_is_n_frames_times_raw_pixels():
    frames = [_solid(16, 8, (i, i, i)) for i in range(5)]
    seq = encode_frame_diff(frames, block_size=8)
    assert frame_diff_full_size(seq) == 5 * 16 * 8 * 3


def test_payload_size_counts_baseline_plus_only_changed_blocks():
    f0 = _solid(16, 16, (0, 0, 0))
    f1 = f0.copy()
    f1[0:8, 0:8] = (1, 1, 1)                # 1 block changes
    f2 = f1.copy()
    f2[8:16, 8:16] = (2, 2, 2)             # 1 more block changes
    seq = encode_frame_diff([f0, f1, f2], block_size=8)

    baseline_bytes = 16 * 16 * 3
    block_bytes = 8 * 8 * 3
    assert frame_diff_changed_blocks(seq) == [1, 1]
    assert frame_diff_payload_size(seq) == baseline_bytes + 2 * block_bytes
    assert frame_diff_payload_size(seq) < frame_diff_full_size(seq)


def test_worst_case_every_block_changing_approaches_full_storage():
    rng = np.random.RandomState(7)
    frames = [rng.randint(0, 256, (16, 16, 3), dtype=np.uint8) for _ in range(4)]
    seq = encode_frame_diff(frames, block_size=8)

    # Every frame touches all four blocks, so the delta payload equals full
    # storage exactly (baseline + 3 frames * 4 blocks == 4 full frames).
    assert frame_diff_changed_blocks(seq) == [4, 4, 4]
    assert frame_diff_payload_size(seq) == frame_diff_full_size(seq)
    # Even the worst case stays lossless.
    rebuilt = decode_frame_diff(seq)
    assert frame_diff_pixel_mismatch(frames, rebuilt) == [0, 0, 0, 0]


def test_sparse_animation_saves_most_of_its_volume():
    # Six frames where only one of sixteen blocks ever blinks.
    f0 = _solid(32, 32, (12, 24, 48))
    frames = [f0]
    for i in range(5):
        nxt = frames[-1].copy()
        nxt[0:8, 0:8] = (200, 200, 100) if i % 2 == 0 else (12, 24, 48)
        frames.append(nxt)
    seq = encode_frame_diff(frames, block_size=8)

    full = frame_diff_full_size(seq)
    payload = frame_diff_payload_size(seq)
    assert payload <= full * 0.25           # >=75% of full storage saved
    assert frame_diff_pixel_mismatch(frames, decode_frame_diff(seq)) == [0] * 6


# --------------------------------------------------------------------------
# Input validation
# --------------------------------------------------------------------------

def test_invalid_arguments_raise_valueerror():
    f0 = _solid(8, 8, (0, 0, 0))
    f1 = f0.copy()
    f1[0, 0] = (1, 1, 1)

    with pytest.raises(ValueError):
        encode_frame_diff([])
    with pytest.raises(ValueError):
        encode_frame_diff([f0, f1], block_size=0)
    with pytest.raises(ValueError):
        encode_frame_diff([f0, f1], block_size=7)       # 8 not a multiple of 7
    with pytest.raises(ValueError):
        encode_frame_diff([f0, f1], change_threshold=-1)
    with pytest.raises(ValueError):
        encode_frame_diff([f0, f1], block_size=8,
                          change_threshold=64)          # >= pixels per block
    with pytest.raises(ValueError):
        encode_frame_diff([f0, np.zeros((4, 4, 3), np.uint8)], block_size=4)
    with pytest.raises(ValueError):
        encode_frame_diff([np.zeros((8, 8), np.uint8)], block_size=8)  # 2-D


def test_decode_rejects_malformed_delta_indices():
    f0 = _solid(8, 8, (0, 0, 0))
    seq = encode_frame_diff([f0, f0], block_size=8)
    # Only one block exists in an 8x8 grid with block_size 8.
    seq.deltas[0] = FrameDiff(
        changed=np.array([9], dtype=np.int32),
        blocks=np.zeros((1, 8, 8, 3), dtype=np.uint8),
    )
    with pytest.raises(ValueError, match="outside"):
        decode_frame_diff(seq)


def test_single_frame_sequence_is_just_the_baseline():
    f0 = _solid(8, 16, (3, 2, 1))
    seq = encode_frame_diff([f0], block_size=8)
    assert seq.deltas == []
    rebuilt = decode_frame_diff(seq)
    assert len(rebuilt) == 1 and np.array_equal(rebuilt[0], f0)
    assert frame_diff_full_size(seq) == frame_diff_payload_size(seq)
