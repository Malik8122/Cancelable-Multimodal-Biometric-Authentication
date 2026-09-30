"""Offline tests for preprocessing/voice.py.

Synthetic sine-wave "speech" only - no VoxCeleb, no network, no optional
dependency (torchaudio/speechbrain/webrtcvad) required - matching this
repo's existing offline-test convention (see tests/conftest.py).
"""

from __future__ import annotations

import numpy as np
import pytest

from preprocessing.voice import N_MELS, VoicePreprocessor, load_wav_file


def _tone(duration_seconds: float, sample_rate: int, frequency: float = 220.0, amplitude: float = 0.5) -> np.ndarray:
    t = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds), endpoint=False)
    return (amplitude * np.sin(2 * np.pi * frequency * t)).astype(np.float32)


def test_mono_conversion_averages_channels():
    preprocessor = VoicePreprocessor()
    left = _tone(1.0, preprocessor.sample_rate, frequency=220.0)
    right = _tone(1.0, preprocessor.sample_rate, frequency=220.0) * 0.0  # silent right channel
    stereo = np.stack([left, right], axis=-1)

    mono = preprocessor._to_mono(stereo)

    assert mono.ndim == 1
    np.testing.assert_allclose(mono, left * 0.5, atol=1e-6)


def test_resample_changes_length_proportionally():
    preprocessor = VoicePreprocessor(sample_rate=16_000)
    original = _tone(1.0, sample_rate=44_100)

    resampled = preprocessor._resample(original, original_sample_rate=44_100)

    assert abs(len(resampled) - 16_000) <= 1


def test_resample_is_a_no_op_when_already_at_target_rate():
    preprocessor = VoicePreprocessor(sample_rate=16_000)
    original = _tone(1.0, sample_rate=16_000)

    resampled = preprocessor._resample(original, original_sample_rate=16_000)

    assert resampled is original


def test_fixed_length_segment_pads_shorter_clips():
    preprocessor = VoicePreprocessor(sample_rate=16_000, clip_seconds=4.0)
    short_clip = _tone(1.0, sample_rate=16_000)

    segment = preprocessor._fixed_length_segment(short_clip, training=False, rng=None)

    assert len(segment) == preprocessor.target_num_samples
    assert np.array_equal(segment[: len(short_clip)], short_clip)
    assert np.all(segment[len(short_clip) :] == 0)


def test_fixed_length_segment_center_crops_longer_clips_deterministically():
    preprocessor = VoicePreprocessor(sample_rate=16_000, clip_seconds=1.0)
    long_clip = _tone(4.0, sample_rate=16_000)

    first = preprocessor._fixed_length_segment(long_clip, training=False, rng=None)
    second = preprocessor._fixed_length_segment(long_clip, training=False, rng=None)

    assert len(first) == preprocessor.target_num_samples
    assert np.array_equal(first, second)


def test_fixed_length_segment_random_crops_during_training():
    preprocessor = VoicePreprocessor(sample_rate=16_000, clip_seconds=1.0)
    long_clip = _tone(4.0, sample_rate=16_000, frequency=1.0)  # slow-varying so crops differ

    crops = [
        preprocessor._fixed_length_segment(long_clip, training=True, rng=np.random.default_rng(seed))
        for seed in range(5)
    ]

    assert all(len(crop) == preprocessor.target_num_samples for crop in crops)
    assert not all(np.array_equal(crops[0], crop) for crop in crops[1:])


def test_trim_silence_removes_a_silent_segment():
    preprocessor = VoicePreprocessor(sample_rate=16_000)
    silence = np.zeros(16_000, dtype=np.float32)
    tone = _tone(1.0, sample_rate=16_000, amplitude=0.8)
    clip = np.concatenate([silence, tone, silence])

    trimmed = preprocessor._trim_silence(clip)

    assert len(trimmed) < len(clip)


def test_trim_silence_on_fully_silent_clip_returns_input_unchanged():
    preprocessor = VoicePreprocessor(sample_rate=16_000)
    silence = np.zeros(16_000, dtype=np.float32)

    trimmed = preprocessor._trim_silence(silence)

    assert np.array_equal(trimmed, silence)


def test_trim_silence_never_returns_more_samples_than_the_input():
    """Regression test for the authentication-debug-sprint bug: `_trim_silence`
    used to concatenate each *voiced* frame directly, but consecutive frames
    overlap (`hop_length=160` < `frame_length=400`), so a mostly-or-fully
    voiced clip - the common case, most real utterances aren't mostly silence -
    got each sample re-emitted once per overlapping voiced frame that covered
    it. A fully-voiced 4-second (64000-sample) clip was measured to come out
    as ~159200 samples: 2.5x longer than the input, entirely made of
    duplicated, shifted copies of itself.

    `_fixed_length_segment`'s center-crop then selected from that unstable,
    duplicated sequence, so a tiny amount of realistic input noise changed
    which content survived the crop and the resulting embedding changed
    catastrophically (observed: genuine same-speaker cosine similarity of
    -0.18 instead of ~1.0) even though the whole pipeline is deterministic.

    "Trimming silence" can only ever *remove* content, never add it - this is
    the invariant that would have caught the bug immediately.
    """
    preprocessor = VoicePreprocessor(sample_rate=16_000)
    fully_voiced = _tone(4.0, sample_rate=16_000, amplitude=0.5)

    trimmed = preprocessor._trim_silence(fully_voiced)

    assert len(trimmed) <= len(fully_voiced)


def test_trim_silence_keeps_every_sample_exactly_once_when_fully_voiced():
    """A stricter version of the invariant above: for a clip with no silence
    at all, trimming should be close to a no-op (every sample is voiced), not
    merely "not longer" - it must not silently drop *or* duplicate content."""
    preprocessor = VoicePreprocessor(sample_rate=16_000)
    fully_voiced = _tone(4.0, sample_rate=16_000, amplitude=0.5)

    trimmed = preprocessor._trim_silence(fully_voiced)

    # A few samples at the very end can legitimately fall outside the last
    # frame window, but the result must be (near-)the full length, not a
    # small fraction of it and not a multiple of it.
    assert len(fully_voiced) - 400 <= len(trimmed) <= len(fully_voiced)


def test_normalize_loudness_scales_toward_target_rms():
    preprocessor = VoicePreprocessor(target_rms=0.1)
    quiet = _tone(1.0, sample_rate=16_000, amplitude=0.01)

    normalized = preprocessor._normalize_loudness(quiet)

    normalized_rms = np.sqrt(np.mean(normalized**2))
    assert normalized_rms == pytest.approx(0.1, abs=1e-3)


def test_normalize_loudness_on_silence_is_a_no_op():
    preprocessor = VoicePreprocessor()
    silence = np.zeros(1000, dtype=np.float32)

    assert np.array_equal(preprocessor._normalize_loudness(silence), silence)


def test_short_clip_mel_padding_does_not_contaminate_real_frames_mean():
    """Regression test for the reliability-sprint bug: `preprocess()` used to
    pad the *waveform* with raw silence before mel extraction whenever the
    trimmed clip fell short of the target length, so `_log_mel_filterbank`'s
    per-bin mean-normalization was computed across a mix of real speech
    frames and artificial near-silent frames (log(1e-10) =~ -23, an extreme
    outlier next to real speech energies) - dragging the mean down and
    shifting the normalized values for the real frames too. The fix computes
    the mel filterbank (and its normalization) on the real, unpadded content
    only, then pads the resulting *frames* with an already-mean-subtracted
    zero - a neutral value, not an outlier.

    This is checked directly: mel-normalizing a short clip on its own (no
    padding at all, by using a preprocessor whose target length matches the
    clip exactly) must equal the real-frame portion of what `preprocess()`
    produces when that same clip is *shorter* than a larger target - if
    padding were still contaminating the mean, these would differ.
    """
    sample_rate = 16_000
    short_clip = _tone(1.0, sample_rate=sample_rate, amplitude=0.5)  # no silence to trim

    reference_preprocessor = VoicePreprocessor(sample_rate=sample_rate, clip_seconds=1.0)
    reference_mel = reference_preprocessor.preprocess(short_clip, sample_rate=sample_rate)

    padded_target_preprocessor = VoicePreprocessor(sample_rate=sample_rate, clip_seconds=4.0)
    padded_mel = padded_target_preprocessor.preprocess(short_clip, sample_rate=sample_rate)

    assert padded_mel.shape[1] == padded_target_preprocessor.target_num_frames
    real_frames = padded_mel[:, : reference_mel.shape[1]]
    np.testing.assert_allclose(real_frames, reference_mel, atol=1e-4)
    # The padded tail must be exactly zero (mean-subtracted neutral value),
    # never a real-looking value that could be mistaken for content.
    assert np.array_equal(padded_mel[:, reference_mel.shape[1] :], np.zeros_like(padded_mel[:, reference_mel.shape[1] :]))


def test_preprocess_returns_expected_mel_shape():
    preprocessor = VoicePreprocessor(sample_rate=16_000, clip_seconds=2.0, n_mels=N_MELS)
    raw_audio = _tone(2.0, sample_rate=16_000)

    mel = preprocessor.preprocess(raw_audio, sample_rate=16_000)

    assert mel.ndim == 2
    assert mel.shape[0] == N_MELS
    assert mel.dtype == np.float32


def test_preprocess_output_is_deterministic_in_inference_mode():
    preprocessor = VoicePreprocessor(sample_rate=16_000, clip_seconds=2.0)
    raw_audio = _tone(3.0, sample_rate=16_000)

    first = preprocessor.preprocess(raw_audio, sample_rate=16_000, training=False)
    second = preprocessor.preprocess(raw_audio, sample_rate=16_000, training=False)

    np.testing.assert_array_equal(first, second)


def test_preprocess_handles_a_different_input_sample_rate():
    preprocessor = VoicePreprocessor(sample_rate=16_000, clip_seconds=1.0)
    raw_audio = _tone(1.0, sample_rate=44_100)

    mel = preprocessor.preprocess(raw_audio, sample_rate=44_100)

    assert mel.shape[0] == N_MELS


def test_load_wav_file_roundtrips_int16(tmp_path):
    from scipy.io import wavfile

    sample_rate = 16_000
    original = (_tone(1.0, sample_rate) * 32767).astype(np.int16)
    wav_path = tmp_path / "sample.wav"
    wavfile.write(wav_path, sample_rate, original)

    waveform, loaded_rate = load_wav_file(wav_path)

    assert loaded_rate == sample_rate
    assert waveform.dtype == np.float32
    assert np.max(np.abs(waveform)) <= 1.0
    np.testing.assert_allclose(waveform, original.astype(np.float32) / 32767.0, atol=1e-4)
