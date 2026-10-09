import re

MAX_WORDS = 32
MAX_WORD_LENGTH = 64
_WORD_PATTERN = re.compile(r"^[\w'\-]+$")


def parse_definition(definition: str) -> list[str]:
    """Parse a Shabda definition such as ``hello,big_world:2`` into unique sample bank names.

    Shabda's ``:n`` variation count is accepted but ignored, as speech is deterministic.
    """
    words: list[str] = []
    for token in definition.replace(" ", "_").split(","):
        word = token.split(":", 1)[0].strip()
        if not word:
            continue
        if len(word) > MAX_WORD_LENGTH or not _WORD_PATTERN.match(word):
            raise ValueError(f"Invalid word: {word!r}")
        if word not in words:
            words.append(word)

    if not words:
        raise ValueError("Definition contains no words")
    if len(words) > MAX_WORDS:
        raise ValueError(f"Definition exceeds {MAX_WORDS} words")
    return words


def spoken_text(word: str) -> str:
    return word.replace("_", " ")
