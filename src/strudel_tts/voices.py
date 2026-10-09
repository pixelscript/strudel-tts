import re
from dataclasses import dataclass
from pathlib import Path

_NAME_PATTERN = re.compile(r"^[\w\-]+$")
_KOKORO_VOICE_PATTERN = re.compile(r"^[a-z]{2}_[a-z]+$")


@dataclass(frozen=True)
class KokoroVoice:
    name: str
    lang_code: str


@dataclass(frozen=True)
class _KokoroLanguage:
    lang_code: str
    female_voice: str
    male_voice: str


_KOKORO_LANGUAGES = {
    "en-us": _KokoroLanguage("a", "af_heart", "am_michael"),
    "en-gb": _KokoroLanguage("b", "bf_emma", "bm_george"),
    "es": _KokoroLanguage("e", "ef_dora", "em_alex"),
    "fr": _KokoroLanguage("f", "ff_siwis", "ff_siwis"),
    "hi": _KokoroLanguage("h", "hf_alpha", "hm_omega"),
    "it": _KokoroLanguage("i", "if_sara", "im_nicola"),
    "pt": _KokoroLanguage("p", "pf_dora", "pm_alex"),
}
_DEFAULT_ENGLISH = "en-gb"


def kokoro_voice_for(language: str, gender: str) -> KokoroVoice:
    """Map Shabda's ``language``/``gender`` query parameters onto a Kokoro preset."""
    normalized = language.lower()
    primary = normalized.split("-", 1)[0]
    kokoro_language = (
        _KOKORO_LANGUAGES.get(normalized)
        or _KOKORO_LANGUAGES.get(primary)
        or (_KOKORO_LANGUAGES[_DEFAULT_ENGLISH] if primary == "en" else None)
    )
    if kokoro_language is None:
        supported = ", ".join(sorted(_KOKORO_LANGUAGES))
        raise ValueError(f"Unsupported language {language!r}; supported: {supported}")

    name = kokoro_language.male_voice if gender.lower() == "m" else kokoro_language.female_voice
    return KokoroVoice(name=name, lang_code=kokoro_language.lang_code)


def kokoro_voice_named(name: str) -> KokoroVoice:
    """Kokoro preset names start with their language code, e.g. ``af_bella`` is American English."""
    if not _KOKORO_VOICE_PATTERN.match(name):
        raise ValueError(f"Invalid Kokoro voice name: {name!r}")
    return KokoroVoice(name=name, lang_code=name[0])


@dataclass(frozen=True)
class RvcVoice:
    name: str
    model_path: Path
    index_path: Path | None


class RvcVoiceLibrary:
    """Discovers RVC voices laid out as ``<root>/<name>/<model>.pth`` with an optional ``.index``."""

    def __init__(self, root: Path):
        self._root = root

    def find(self, name: str) -> RvcVoice:
        if not _NAME_PATTERN.match(name):
            raise ValueError(f"Invalid voice name: {name!r}")

        voice_dir = self._root / name
        model_path = _first_match(voice_dir, "*.pth") or _first_match(voice_dir, "*.safetensors")
        if model_path is None:
            raise LookupError(f"No RVC model found for voice {name!r} in {voice_dir}")
        return RvcVoice(name=name, model_path=model_path, index_path=_first_match(voice_dir, "*.index"))

    def available(self) -> list[RvcVoice]:
        voices = []
        for voice_dir in sorted(self._root.glob("*/")):
            try:
                voices.append(self.find(voice_dir.name))
            except (LookupError, ValueError):
                continue
        return voices


def _first_match(directory: Path, pattern: str) -> Path | None:
    return next(iter(sorted(directory.glob(pattern))), None) if directory.is_dir() else None
