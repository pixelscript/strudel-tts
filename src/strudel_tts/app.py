from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .definition import parse_definition
from .speech import SpeechSettings, SpeechSynthesizer
from .voices import RvcVoiceLibrary, kokoro_voice_for, kokoro_voice_named

SAMPLES_ROUTE = "speech_samples"


def create_app(samples_dir: Path, voices_dir: Path, kokoro_model_id: str) -> FastAPI:
    samples_dir.mkdir(parents=True, exist_ok=True)
    voice_library = RvcVoiceLibrary(voices_dir)
    state: dict[str, SpeechSynthesizer] = {}

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        synthesizer = SpeechSynthesizer(samples_dir, kokoro_model_id)
        await run_in_threadpool(synthesizer.warm_up, kokoro_voice_for("en-GB", "f"), voice_library.available())
        state["synthesizer"] = synthesizer
        yield

    app = FastAPI(title="strudel-tts", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"])

    @app.get("/speech/{definition}.json")
    async def speech_json(
        request: Request,
        definition: str,
        gender: str = "f",
        language: str = "en-GB",
        # Strudel's shabda/speech shortcut sends `strudel=1'` (stray quote), so accept any non-empty value.
        strudel: str = "",
        voice: str | None = Query(None, description="RVC voice folder name under the voices directory"),
        kokoro: str | None = Query(None, description="Kokoro preset, e.g. af_bella; overrides gender/language"),
        pitch: int = Query(0, ge=-24, le=24, description="RVC pitch shift in semitones"),
        speed: float = Query(1.0, ge=0.5, le=2.0),
    ):
        try:
            words = parse_definition(definition)
            settings = SpeechSettings(
                kokoro_voice=kokoro_voice_named(kokoro) if kokoro else kokoro_voice_for(language, gender),
                speed=speed,
                rvc_voice=voice_library.find(voice) if voice else None,
                pitch_shift=pitch,
            )
        except LookupError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

        synthesizer = state["synthesizer"]
        for word in words:
            await run_in_threadpool(synthesizer.ensure_sample, word, settings)

        sample_urls = {
            word: f"{SAMPLES_ROUTE}/{quote(synthesizer.relative_sample_path(word, settings).as_posix())}"
            for word in words
        }
        if strudel:
            return {"_base": str(request.base_url), **{word: [url] for word, url in sample_urls.items()}}
        return [
            {"url": url, "type": "audio", "bank": word, "n": 0}
            for word, url in sample_urls.items()
        ]

    app.mount(f"/{SAMPLES_ROUTE}", StaticFiles(directory=samples_dir), name=SAMPLES_ROUTE)
    return app
