"""Audio Transcript Service API: upload audio, get its text back from Whisper running on ONNX Runtime."""

import asyncio
import hmac
import logging
import shutil
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse

from app.asr import OnnxWhisperTranscriber, Transcriber
from app.audio import MEDIA_TYPES, InvalidAudioError, decode_audio, detect_format, duration_seconds
from app.benchmark import (
    VARIANTS,
    BenchmarkError,
    BenchmarkRunner,
    BenchmarkStore,
    NotFound,
    cut_interval,
)
from app.config import Settings, get_settings
from app.formats import RENDERERS
from app.jobs import JobNotFound, QueueFullError, TranscriptionService
from app.schemas import (
    BenchmarkCreate,
    BenchmarkModel,
    BenchmarkRun,
    BenchmarkSource,
    BenchmarkVariant,
    ClipCreate,
    ClipPatch,
    HealthResponse,
    Job,
    JobStatus,
    JobSummary,
    OutputFormat,
    ReferenceClipInfo,
    TranscriptResponse,
)

logger = logging.getLogger(__name__)

UPLOAD_CHUNK = 1024 * 1024
MULTIPART_SLACK = 1024 * 1024  # form fields + boundaries on top of the file itself
STATIC_DIR = Path(__file__).parent / "static"
UPLOAD_PATHS = ("/transcribe", "/jobs", "/benchmark/sources", "/benchmark/clips/upload")
# Guarded when API_TOKEN is set. The web page, health checks and the API docs stay reachable.
PROTECTED_PREFIXES = ("/transcribe", "/jobs", "/languages", "/benchmark", "/benchmarks", "/config")
LANGUAGE_PATTERN = r"^([a-z]{2,3}|auto)$"


def _is_protected(path: str) -> bool:
    return any(path == p or path.startswith(p + "/") for p in PROTECTED_PREFIXES)


def _token_ok(request: Request, expected: str) -> bool:
    """Accept `X-API-Key: <token>` or `Authorization: Bearer <token>` (scheme case-insensitive)."""
    scheme, _, credentials = request.headers.get("authorization", "").partition(" ")
    bearer = credentials.strip() if scheme.lower() == "bearer" else ""
    supplied = request.headers.get("x-api-key", "").strip() or bearer
    return hmac.compare_digest(supplied.encode(), expected.encode())


@dataclass
class Components:
    """Injectable dependencies; tests pass fakes, production builds them from Settings."""

    transcriber: Transcriber | None = None
    # Loads another model for the benchmark (tests pass a fake); default: OnnxWhisperTranscriber.from_dir.
    load_model: Callable[[Path], Transcriber] | None = None


def _clock(seconds: float) -> str:
    minutes, secs = divmod(seconds, 60)
    return f"{int(minutes)}:{secs:04.1f}"


def create_app(settings: Settings | None = None, components: Components | None = None) -> FastAPI:
    settings = settings or get_settings()
    components = components or Components()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        transcriber, transcriber_error = components.transcriber, None
        if transcriber is None:
            try:
                transcriber = OnnxWhisperTranscriber.from_dir(
                    settings.whisper_model_dir, settings.max_new_tokens, settings.onnx_threads,
                    low_memory=settings.whisper_low_memory,
                )
                logger.info("Loaded speech model %s", transcriber.name)
            except Exception as e:  # keep serving: /health reports the reason, /healthz fails
                transcriber_error = str(e)
                logger.error("Speech model not loaded: %s", e)

        service = TranscriptionService(
            transcriber, settings.upload_dir, transcriber_error,
            max_audio_seconds=settings.max_audio_seconds,
            concurrency=settings.worker_concurrency,
            max_queue_size=settings.max_queue_size,
            max_finished_jobs=settings.max_finished_jobs,
        )
        app.state.service = service
        await service.start()

        def load_model(path: Path) -> Transcriber:
            return OnnxWhisperTranscriber.from_dir(str(path), settings.max_new_tokens, settings.onnx_threads,
                                                   low_memory=settings.whisper_low_memory)

        store = BenchmarkStore(settings.benchmark_dir)
        store.setup()
        app.state.benchmark = BenchmarkRunner(
            store, service.submit_task,
            (Path(settings.whisper_model_dir).name, transcriber) if transcriber else None,
            settings.models_dir, components.load_model or load_model,
        )
        app.state.benchmark.recover()
        yield
        await service.stop()

    app = FastAPI(
        title="Audio Transcript Service",
        description="Transcribe audio to text with Whisper on ONNX Runtime: plain text, timestamped "
                    "segments, SRT or WebVTT. The benchmark scores models and decoding options by word "
                    "error rate on reference intervals of your own recordings.",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def guard(request: Request, call_next):
        """Reject unauthenticated or oversized requests before the request body is read or spooled."""
        method, path = request.method, request.url.path.rstrip("/")
        if settings.api_token and _is_protected(path) and not _token_ok(request, settings.api_token):
            return JSONResponse({"detail": "Missing or invalid API token"}, status_code=401,
                                headers={"WWW-Authenticate": "Bearer"})
        if method == "POST" and path in UPLOAD_PATHS:
            declared = request.headers.get("content-length")
            if declared and declared.isdigit() and int(declared) > settings.max_upload_bytes + MULTIPART_SLACK:
                return JSONResponse({"detail": f"File exceeds {settings.max_upload_mb} MB limit"}, status_code=413)
        return await call_next(request)

    def service_of(request: Request) -> TranscriptionService:
        return request.app.state.service

    def get_job(service: TranscriptionService, job_id: str) -> Job:
        try:
            return service.get(job_id)
        except JobNotFound:
            raise HTTPException(404, f"Job {job_id} not found")

    def check_precision(service: TranscriptionService, encoder: str | None, decoder: str | None) -> None:
        available = getattr(service.transcriber, "available_precisions", None) or {}
        default = getattr(service.transcriber, "precision", None) or {}
        for part, p in (("encoder", encoder), ("decoder", decoder)):
            if p and p != default.get(part) and p not in available.get(part, []):
                have = ", ".join(available.get(part) or [default.get(part) or "the default only"])
                raise HTTPException(422, f"{part} precision {p} isn't available in this export ({have}); see GET /config")

    def check_ready(service: TranscriptionService, language: str | None) -> str:
        """Fail early when nothing could be transcribed; return the language to use."""
        if service.transcriber is None:
            raise HTTPException(503, f"Speech model unavailable: {service.transcriber_error or 'not loaded'}")
        language = language or settings.default_language
        if language != "auto" and service.languages and language not in service.languages:
            raise HTTPException(422, f"Unsupported language '{language}'; see GET /languages")
        return language

    async def save_upload(file: UploadFile, path: Path) -> int:
        """Stream an upload to `path`, checking its format and size. Returns the size in bytes."""
        size = 0
        try:
            with path.open("wb") as out:
                while chunk := await file.read(UPLOAD_CHUNK):
                    if size == 0:
                        detect_format(chunk)
                    size += len(chunk)
                    if size > settings.max_upload_bytes:
                        raise HTTPException(413, f"File exceeds {settings.max_upload_mb} MB limit")
                    out.write(chunk)
            if size == 0:
                raise HTTPException(400, "Empty file")
        except InvalidAudioError as e:
            path.unlink(missing_ok=True)
            raise HTTPException(415, str(e))
        except HTTPException:
            path.unlink(missing_ok=True)
            raise
        return size

    def submit_job(service: TranscriptionService, job_id: str, filename: str, size: int, language: str,
                   temperature_fallback: bool | None = None, condition_on_previous_text: bool | None = None,
                   initial_prompt: str | None = None, encoder: str | None = None,
                   decoder: str | None = None) -> Job:
        """Queue the audio at service.upload_path(job_id), with the default decoding options unless given."""
        job = Job(
            id=job_id, filename=filename, size_bytes=size, language=None if language == "auto" else language,
            temperature_fallback=(settings.temperature_fallback if temperature_fallback is None
                                  else temperature_fallback),
            condition_on_previous_text=(settings.condition_on_previous_text if condition_on_previous_text is None
                                        else condition_on_previous_text),
            initial_prompt=(initial_prompt or settings.initial_prompt or "").strip() or None,
            encoder_precision=encoder, decoder_precision=decoder,
        )
        try:
            service.submit(job)
        except QueueFullError as e:
            raise HTTPException(503, str(e), headers={"Retry-After": "30"})
        return job

    async def accept_upload(service: TranscriptionService, file: UploadFile, language: str | None,
                            **options) -> Job:
        """Validate and store the upload, then queue it as a job."""
        language = check_ready(service, language)
        check_precision(service, options.get("encoder"), options.get("decoder"))
        job_id = uuid.uuid4().hex
        size = await save_upload(file, service.upload_path(job_id))
        return submit_job(service, job_id, file.filename or "audio", size, language, **options)

    def render(job: Job, fmt: OutputFormat):
        if fmt == OutputFormat.json:
            return TranscriptResponse(**job.transcript.model_dump(), model=job.model, precision=job.precision,
                                      metrics=job.metrics)
        renderer, media_type = RENDERERS[fmt.value]
        stem = Path(job.filename).stem or "transcript"
        extension = "txt" if fmt == OutputFormat.text else fmt.value
        return PlainTextResponse(renderer(job.transcript), media_type=media_type,
                                 headers={"Content-Disposition": f'inline; filename="{stem}.{extension}"'})

    def summary(job: Job) -> JobSummary:
        return JobSummary(id=job.id, filename=job.filename, status=job.status, status_url=f"/jobs/{job.id}",
                          transcript_url=f"/jobs/{job.id}/transcript")

    file_param = File(..., description="WAV, MP3, OGG/Opus, M4A or FLAC audio")
    language_param = Form(None, pattern=LANGUAGE_PATTERN,
                          description="ISO code such as en, de, fr, it; 'auto' to detect")
    format_param = Query(OutputFormat.json, description="json (segments + metadata), text, srt or vtt")
    fallback_param = Form(None, description="Re-decode windows whose text repeats or is unsure "
                                            "(default: TEMPERATURE_FALLBACK)")
    previous_param = Form(None, description="Prompt each 30 s window with the text before it "
                                            "(default: CONDITION_ON_PREVIOUS_TEXT)")
    prompt_param = Form(None, max_length=1000, description="Vocabulary or style hint, e.g. names and terms")
    encoder_param = Form(None, pattern=r"^(fp32|int8)$",
                         description="Encoder precision: fp32 (full, the default) or int8 (smaller, less accurate)")
    decoder_param = Form(None, pattern=r"^(fp32|int8)$",
                         description="Decoder precision: int8 (the default) or fp32 (full)")

    # ------------------------------------------------------------ synchronous

    @app.post("/transcribe", response_model=TranscriptResponse,
              responses={200: {"content": {"text/plain": {}, "application/x-subrip": {}, "text/vtt": {}}}})
    async def transcribe(request: Request, file: UploadFile = file_param, language: str | None = language_param,
                         format: OutputFormat = format_param, temperature_fallback: bool | None = fallback_param,
                         condition_on_previous_text: bool | None = previous_param,
                         initial_prompt: str | None = prompt_param, encoder: str | None = encoder_param,
                         decoder: str | None = decoder_param):
        """Upload audio and wait for its transcript. Shares the job queue, so it waits its turn."""
        service = service_of(request)
        job = await accept_upload(service, file, language, temperature_fallback=temperature_fallback,
                                  condition_on_previous_text=condition_on_previous_text,
                                  initial_prompt=initial_prompt, encoder=encoder, decoder=decoder)
        try:
            job = await service.wait(job.id)
        finally:
            if job.id in service.jobs:
                service.delete(job.id)  # the caller has the result; nothing to poll for
        if job.status != JobStatus.completed:
            raise HTTPException(422, f"Transcription failed: {job.error}")
        return render(job, format)

    # ------------------------------------------------------------------- jobs

    @app.post("/jobs", response_model=JobSummary, status_code=202)
    async def create_job(request: Request, file: UploadFile = file_param, language: str | None = language_param,
                         temperature_fallback: bool | None = fallback_param,
                         condition_on_previous_text: bool | None = previous_param,
                         initial_prompt: str | None = prompt_param, encoder: str | None = encoder_param,
                         decoder: str | None = decoder_param):
        """Upload audio for background transcription; poll the status URL, then fetch the transcript.

        `encoder` / `decoder` pick the precision of each part for this job (see GET /config for what the
        export has); a non-default part is loaded for the job and released after it."""
        job = await accept_upload(service_of(request), file, language, temperature_fallback=temperature_fallback,
                                  condition_on_previous_text=condition_on_previous_text,
                                  initial_prompt=initial_prompt, encoder=encoder, decoder=decoder)
        return summary(job)

    @app.get("/jobs", response_model=list[JobSummary])
    async def list_jobs(request: Request):
        return [summary(j) for j in service_of(request).list()]

    @app.get("/jobs/{job_id}", response_model=Job, response_model_exclude={"transcript"})
    async def job_status(request: Request, job_id: str):
        return get_job(service_of(request), job_id)

    @app.get("/jobs/{job_id}/transcript", response_model=TranscriptResponse,
             responses={200: {"content": {"text/plain": {}, "application/x-subrip": {}, "text/vtt": {}}}})
    async def job_transcript(request: Request, job_id: str, format: OutputFormat = format_param):
        job = get_job(service_of(request), job_id)
        if job.status == JobStatus.failed:
            raise HTTPException(422, f"Transcription failed: {job.error}")
        if job.status != JobStatus.completed:
            raise HTTPException(409, f"Transcript not ready (status: {job.status.value})")
        return render(job, format)

    @app.delete("/jobs/{job_id}", status_code=204)
    async def delete_job(request: Request, job_id: str):
        service = service_of(request)
        get_job(service, job_id)
        service.delete(job_id)

    # -------------------------------------------------------------- benchmark
    # Reference set: intervals of real recordings with their correct transcripts. A run transcribes them
    # with each chosen model and decoding variant and reports the word error rate.

    def runner_of(request: Request) -> BenchmarkRunner:
        return request.app.state.benchmark

    def store_of(request: Request) -> BenchmarkStore:
        return runner_of(request).store

    def found(lookup, *args):
        try:
            return lookup(*args)
        except NotFound:
            raise HTTPException(404, f"{args[0]} not found")

    def refresh_source(request: Request, source: BenchmarkSource) -> BenchmarkSource:
        """Copy the proposal job's outcome into the stored recording once it has one."""
        if source.status in (JobStatus.completed, JobStatus.failed) or source.job_id is None:
            return source
        try:
            job = service_of(request).get(source.job_id)
        except JobNotFound:  # jobs live in memory: a restart lost it
            source.status, source.error = JobStatus.failed, "Interrupted by a restart: transcribe it again"
        else:
            source.status, source.error = job.status, job.error
            if job.status == JobStatus.completed:
                source.segments = job.transcript.segments
                source.duration_seconds = job.transcript.duration_seconds
                source.language = job.transcript.language
        if source.status in (JobStatus.completed, JobStatus.failed):
            store_of(request).save_source(source)
        return source

    def propose_segments(request: Request, source: BenchmarkSource, language: str) -> None:
        """Transcribe a copy of the recording (jobs delete their upload) to propose intervals."""
        service = service_of(request)
        job_id = uuid.uuid4().hex
        shutil.copyfile(store_of(request).source_audio(source.id), service.upload_path(job_id))
        try:
            submit_job(service, job_id, source.filename, source.size_bytes, language)
        except HTTPException:
            service.upload_path(job_id).unlink(missing_ok=True)
            raise
        source.job_id, source.status, source.error, source.segments = job_id, JobStatus.queued, None, None
        store_of(request).save_source(source)

    @app.get("/benchmark/models", response_model=list[BenchmarkModel])
    async def benchmark_models(request: Request):
        """The service's model plus every export found in MODELS_DIR."""
        return runner_of(request).models()

    @app.get("/benchmark/variants", response_model=list[BenchmarkVariant])
    async def benchmark_variants():
        return list(VARIANTS.values())

    @app.post("/benchmark/sources", response_model=BenchmarkSource, status_code=202)
    async def add_source(request: Request, file: UploadFile = file_param, language: str | None = language_param):
        """Upload a real recording. Whisper transcribes it to propose intervals to pick and correct."""
        language = check_ready(service_of(request), language)
        store = store_of(request)
        source_id = uuid.uuid4().hex
        path = store.source_audio(source_id)
        size = await save_upload(file, path)
        with path.open("rb") as f:
            media_type = MEDIA_TYPES[detect_format(f.read(16))]
        source = BenchmarkSource(id=source_id, filename=file.filename or "recording", size_bytes=size,
                                 media_type=media_type, language=None if language == "auto" else language)
        try:
            propose_segments(request, source, language)
        except HTTPException:
            path.unlink(missing_ok=True)
            (store.sources_dir / f"{source_id}.json").unlink(missing_ok=True)
            raise
        return source

    @app.get("/benchmark/sources", response_model=list[BenchmarkSource],
             response_model_exclude={"__all__": {"segments"}})
    async def list_sources(request: Request):
        return [refresh_source(request, s) for s in store_of(request).list_sources()]

    @app.get("/benchmark/sources/{source_id}", response_model=BenchmarkSource)
    async def get_source(request: Request, source_id: str):
        return refresh_source(request, found(store_of(request).get_source, source_id))

    @app.get("/benchmark/sources/{source_id}/audio")
    async def source_audio(request: Request, source_id: str):
        source = found(store_of(request).get_source, source_id)
        return FileResponse(store_of(request).source_audio(source_id), media_type=source.media_type)

    @app.post("/benchmark/sources/{source_id}/transcribe", response_model=BenchmarkSource, status_code=202)
    async def retranscribe_source(request: Request, source_id: str,
                                  language: str | None = Query(None, pattern=LANGUAGE_PATTERN)):
        """Propose intervals again, e.g. in another language."""
        source = refresh_source(request, found(store_of(request).get_source, source_id))
        if source.status in (JobStatus.queued, JobStatus.processing):
            raise HTTPException(409, "This recording is already being transcribed")
        propose_segments(request, source, check_ready(service_of(request), language))
        return source

    @app.delete("/benchmark/sources/{source_id}", status_code=204)
    async def delete_source(request: Request, source_id: str):
        """Delete a recording. Clips cut from it stay in the benchmark set."""
        found(store_of(request).delete_source, source_id)

    @app.get("/benchmark/clips", response_model=list[ReferenceClipInfo])
    async def list_clips(request: Request):
        return store_of(request).list_clips()

    @app.post("/benchmark/clips", response_model=ReferenceClipInfo, status_code=201)
    async def create_clip(request: Request, body: ClipCreate):
        """Save an interval of a recording, with its correct transcript, as a reference clip."""
        store = store_of(request)
        source = found(store.get_source, body.source_id)
        if body.end <= body.start:
            raise HTTPException(422, "The interval must end after it starts")
        try:
            samples = await asyncio.to_thread(cut_interval, store.source_audio(source.id), body.start, body.end,
                                              settings.max_clip_seconds)
        except (BenchmarkError, InvalidAudioError) as e:
            raise HTTPException(422, str(e))
        info = ReferenceClipInfo(
            id=uuid.uuid4().hex, name=(body.name or "").strip() or f"{source.filename} {_clock(body.start)}",
            reference=body.reference.strip(), duration_seconds=round(duration_seconds(samples), 2),
            source_id=source.id, source_filename=source.filename,
            start=round(body.start, 2), end=round(body.start + duration_seconds(samples), 2),
        )
        return store.add_clip(info, samples)

    @app.post("/benchmark/clips/upload", response_model=ReferenceClipInfo, status_code=201)
    async def upload_clip(request: Request, file: UploadFile = file_param,
                          reference: str = Form(..., min_length=1, max_length=20_000),
                          name: str | None = Form(None, max_length=200)):
        """Add a clip you already cut, with its correct transcript, as a whole."""
        store = store_of(request)
        tmp = service_of(request).upload_path("clip-" + uuid.uuid4().hex)
        await save_upload(file, tmp)
        try:
            samples = await asyncio.to_thread(decode_audio, tmp, settings.max_clip_seconds, True)
        except InvalidAudioError as e:
            raise HTTPException(415, str(e))
        finally:
            tmp.unlink(missing_ok=True)
        if duration_seconds(samples) > settings.max_clip_seconds:
            raise HTTPException(422, f"A clip can be at most {settings.max_clip_seconds}s long")
        info = ReferenceClipInfo(id=uuid.uuid4().hex, name=(name or "").strip() or Path(file.filename or "clip").stem,
                                 reference=reference.strip(), duration_seconds=round(duration_seconds(samples), 2))
        return store.add_clip(info, samples)

    @app.patch("/benchmark/clips/{clip_id}", response_model=ReferenceClipInfo)
    async def edit_clip(request: Request, clip_id: str, body: ClipPatch):
        return found(store_of(request).update_clip, clip_id, body.reference, body.name)

    @app.delete("/benchmark/clips/{clip_id}", status_code=204)
    async def delete_clip(request: Request, clip_id: str):
        found(store_of(request).delete_clip, clip_id)

    @app.get("/benchmark/clips/{clip_id}/audio")
    async def clip_audio(request: Request, clip_id: str):
        found(store_of(request).get_clip, clip_id)
        return FileResponse(store_of(request).clip_audio(clip_id), media_type="audio/wav")

    @app.post("/benchmarks", response_model=BenchmarkRun, status_code=202)
    async def create_benchmark(request: Request, body: BenchmarkCreate):
        """Queue a run: every clip, with each model and decoding variant. It shares the transcription queue."""
        run = BenchmarkRun(id=uuid.uuid4().hex, models=list(dict.fromkeys(body.models)),
                           variants=list(dict.fromkeys(body.variants)), language=body.language,
                           initial_prompt=(body.initial_prompt or "").strip() or None, clip_ids=body.clip_ids or [])
        try:
            return runner_of(request).create(run)
        except BenchmarkError as e:
            raise HTTPException(422, str(e))
        except QueueFullError as e:
            raise HTTPException(503, str(e), headers={"Retry-After": "30"})

    @app.get("/benchmarks", response_model=list[BenchmarkRun],
             response_model_exclude={"__all__": {"results": {"__all__": {"clips"}}}})
    async def list_benchmarks(request: Request):
        """Runs, newest first, without the per-clip details."""
        return runner_of(request).all_runs()

    @app.get("/benchmarks/{run_id}", response_model=BenchmarkRun)
    async def get_benchmark(request: Request, run_id: str):
        return found(runner_of(request).get, run_id)

    @app.delete("/benchmarks/{run_id}", status_code=204)
    async def delete_benchmark(request: Request, run_id: str):
        runner = runner_of(request)
        if run_id in runner.active:
            raise HTTPException(409, "The run hasn't finished yet")
        found(runner.store.delete_run, run_id)

    # ----------------------------------------------------------------- status

    @app.get("/config")
    async def config(request: Request):
        """How audio becomes text here, with the values actually in use: for the client's Settings tab.

        Audio, window and feature values are fixed by Whisper (the model was trained on exactly these);
        the model, its precision and the decoding defaults are this deployment's choices.
        """
        from app import asr, features

        service = service_of(request)
        t = service.transcriber
        n_mels = getattr(t, "n_mels", 80)
        return {
            "model": {"name": t.name if t else None, "error": service.transcriber_error,
                      "precision": getattr(t, "precision", None),
                      "available_precisions": getattr(t, "available_precisions", None) or {},
                      "low_memory": settings.whisper_low_memory,
                      "onnx_threads": settings.onnx_threads, "runtime": "ONNX Runtime (CPU)"},
            "audio": {"decoder": "ffmpeg (pure-Python fallback for WAV)", "sample_rate": features.SAMPLE_RATE,
                      "channels": 1, "sample_format": "16-bit PCM", "max_audio_seconds": settings.max_audio_seconds,
                      "max_upload_mb": settings.max_upload_mb},
            "windows": {"seconds": features.CHUNK_SECONDS, "samples": features.N_SAMPLES,
                        "conversion": "16-bit → float32 per window, padded with silence or cut to the window"},
            "features": {"n_fft": features.N_FFT, "frame_ms": features.N_FFT * 1000 / features.SAMPLE_RATE,
                         "hop_length": features.HOP_LENGTH,
                         "hop_ms": features.HOP_LENGTH * 1000 / features.SAMPLE_RATE, "window": "periodic Hann",
                         "n_mels": n_mels, "frames": features.N_FRAMES, "shape": [n_mels, features.N_FRAMES],
                         "scale": "log10, floor at (max − 8), then (x + 4) / 4"},
            "decoding": {"default_language": settings.default_language, "max_new_tokens": settings.max_new_tokens,
                         "temperature_fallback": settings.temperature_fallback,
                         "temperatures": list(asr.FALLBACK_TEMPERATURES),
                         "compression_ratio_threshold": asr.COMPRESSION_RATIO_THRESHOLD,
                         "logprob_threshold": asr.LOGPROB_THRESHOLD, "no_speech_threshold": asr.NO_SPEECH_THRESHOLD,
                         "condition_on_previous_text": settings.condition_on_previous_text,
                         "initial_prompt": settings.initial_prompt,
                         "timestamp_precision_seconds": asr.TIME_PRECISION},
        }

    @app.get("/languages")
    async def languages(request: Request):
        """Language codes the loaded model can transcribe."""
        service = service_of(request)
        return {"languages": service.languages, "default": settings.default_language}

    @app.get("/health", response_model=HealthResponse)
    async def health(request: Request):
        service = service_of(request)
        return HealthResponse(
            status="ok" if service.transcriber else "degraded",
            model=service.transcriber.name if service.transcriber else None,
            model_error=service.transcriber_error,
            languages=service.languages,
            queue_depth=service.queue.qsize(),
            active_jobs=service.active,
        )

    @app.get("/healthz", include_in_schema=False)
    async def healthz(request: Request):
        """Liveness for the platform health check: fails if the speech model is missing."""
        if service_of(request).transcriber is None:
            raise HTTPException(503, "Speech model not loaded")
        return {"ok": True}

    @app.get("/", include_in_schema=False)
    async def web_page():
        # no-cache: a deploy's new page shows up on the next load.
        return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})

    return app
