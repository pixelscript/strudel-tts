import os
import tempfile
from pathlib import Path

import numpy as np
import soundfile

SILENCE_THRESHOLD_DB = -45.0
EDGE_PADDING_SECONDS = 0.005
PEAK_DB = -1.0


def trim_silence(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Remove leading/trailing silence so samples start on the beat."""
    threshold = 10 ** (SILENCE_THRESHOLD_DB / 20) * np.max(np.abs(audio), initial=0.0)
    audible = np.flatnonzero(np.abs(audio) > threshold)
    if audible.size == 0:
        return audio

    padding = int(EDGE_PADDING_SECONDS * sample_rate)
    start = max(audible[0] - padding, 0)
    end = min(audible[-1] + padding + 1, audio.size)
    return audio[start:end]


def normalize_peak(audio: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(audio), initial=0.0)
    if peak == 0:
        return audio
    return audio * (10 ** (PEAK_DB / 20) / peak)


def write_wav_atomically(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    """Write via a temp file so concurrent readers never see a partial WAV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temp_name = tempfile.mkstemp(suffix=".wav", dir=path.parent)
    os.close(file_descriptor)
    try:
        soundfile.write(temp_name, audio, sample_rate, subtype="PCM_16")
        os.replace(temp_name, path)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = soundfile.read(path, dtype="float32", always_2d=False)
    return audio, sample_rate
