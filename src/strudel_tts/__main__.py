import argparse
from pathlib import Path

import uvicorn

from .app import create_app

DEFAULT_KOKORO_MODEL = "mlx-community/Kokoro-82M-bf16"


def main() -> None:
    parser = argparse.ArgumentParser(description="Shabda-compatible local speech sample server for Strudel")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--voices-dir", type=Path, default=Path("voices"))
    parser.add_argument("--samples-dir", type=Path, default=Path("speech_samples"))
    parser.add_argument("--kokoro-model", default=DEFAULT_KOKORO_MODEL)
    args = parser.parse_args()

    app = create_app(args.samples_dir.resolve(), args.voices_dir.resolve(), args.kokoro_model)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
