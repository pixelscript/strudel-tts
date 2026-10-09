import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .audio import normalize_peak, read_wav, trim_silence, write_wav_atomically
from .definition import spoken_text
from .voices import KokoroVoice, RvcVoice

RVC_F0_METHOD = "rmvpe"
RVC_INDEX_RATE = 0.5


@dataclass(frozen=True)
class SpeechSettings:
    kokoro_voice: KokoroVoice
    speed: float
    rvc_voice: RvcVoice | None
    pitch_shift: int

    @property
    def cache_key(self) -> str:
        key = f"{self.kokoro_voice.name}_s{self.speed:g}"
        if self.rvc_voice is not None:
            key += f"__{self.rvc_voice.name}_p{self.pitch_shift:+d}"
        return key


class SpeechSynthesizer:
    """Renders words with Kokoro, optionally re-voices them with RVC, and caches the WAVs on disk."""

    def __init__(self, samples_dir: Path, kokoro_model_id: str):
        from mlx_audio.tts.utils import load_model

        self._samples_dir = samples_dir
        self._kokoro = load_model(kokoro_model_id)
        self._kokoro_sample_rate: int = self._kokoro.sample_rate
        self._rvc_pipelines: dict[Path, object] = {}
        # MLX models are not safe to drive from several threads at once.
        self._model_lock = threading.Lock()

    def warm_up(self, voice: KokoroVoice, rvc_voices: list[RvcVoice]) -> None:
        """Load every model up front so the first live request doesn't stall."""
        with self._model_lock:
            audio = self._speak("ready", voice, speed=1.0)
            for rvc_voice in rvc_voices:
                self._convert_voice(audio, rvc_voice, pitch_shift=0)

    def relative_sample_path(self, word: str, settings: SpeechSettings) -> Path:
        return Path(settings.cache_key) / f"{word}.wav"

    def ensure_sample(self, word: str, settings: SpeechSettings) -> Path:
        path = self._samples_dir / self.relative_sample_path(word, settings)
        if path.exists():
            return path

        with self._model_lock:
            if not path.exists():
                audio, sample_rate = self._render(spoken_text(word), settings)
                write_wav_atomically(path, audio, sample_rate)
        return path

    def _render(self, text: str, settings: SpeechSettings) -> tuple[np.ndarray, int]:
        audio = self._speak(text, settings.kokoro_voice, settings.speed)
        sample_rate = self._kokoro_sample_rate
        if settings.rvc_voice is not None:
            audio, sample_rate = self._convert_voice(audio, settings.rvc_voice, settings.pitch_shift)
        return normalize_peak(trim_silence(audio, sample_rate)), sample_rate

    def _speak(self, text: str, voice: KokoroVoice, speed: float) -> np.ndarray:
        segments = [
            np.asarray(result.audio, dtype=np.float32)
            for result in self._kokoro.generate(
                text=text, voice=voice.name, speed=speed, lang_code=voice.lang_code
            )
        ]
        if not segments:
            raise RuntimeError(f"Kokoro produced no audio for {text!r}")
        return np.concatenate(segments)

    def _convert_voice(self, audio: np.ndarray, voice: RvcVoice, pitch_shift: int) -> tuple[np.ndarray, int]:
        pipeline = self._rvc_pipeline(voice.model_path)
        with tempfile.TemporaryDirectory() as work_dir:
            source_path = Path(work_dir) / "source.wav"
            converted_path = Path(work_dir) / "converted.wav"
            write_wav_atomically(source_path, audio, self._kokoro_sample_rate)
            pipeline.convert(
                input_path=source_path,
                output_path=converted_path,
                f0_shift=pitch_shift,
                f0_method=RVC_F0_METHOD,
                index_path=voice.index_path,
                index_rate=RVC_INDEX_RATE,
            )
            return read_wav(converted_path)

    def _rvc_pipeline(self, model_path: Path):
        if model_path not in self._rvc_pipelines:
            from mlx_rvc import RVCPipeline

            self._rvc_pipelines[model_path] = RVCPipeline.from_pretrained(model_path)
        return self._rvc_pipelines[model_path]
