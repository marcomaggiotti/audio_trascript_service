"""API models."""

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class JobStatus(str, Enum):
    queued = "queued"
    processing = "processing"
    completed = "completed"
    failed = "failed"


class OutputFormat(str, Enum):
    json = "json"
    text = "text"
    srt = "srt"
    vtt = "vtt"


class TranscriptSegment(BaseModel):
    index: int
    start: float = Field(description="Seconds from the start of the audio")
    end: float
    text: str
    avg_logprob: float | None = Field(None, description="Mean log-probability of the segment's tokens")
    no_speech_prob: float | None = Field(None, description="Whisper's estimate that the 30 s window is silence")


class InferenceMetrics(BaseModel):
    audio_seconds: float
    decode_seconds: float
    inference_seconds: float
    real_time_factor: float = Field(description="Inference time / audio length; below 1 is faster than real time")
    peak_memory_mb: float


class Transcript(BaseModel):
    language: str | None
    duration_seconds: float
    text: str
    segments: list[TranscriptSegment]


class Job(BaseModel):
    id: str
    filename: str
    size_bytes: int
    language: str | None = Field(None, description="Requested language; None = auto-detect")
    status: JobStatus = JobStatus.queued
    stage: str | None = Field(None, description="While processing: decoding, then transcribing")
    progress: float = Field(0.0, ge=0, le=1, description="Share of the audio transcribed so far (0-1)")
    queue_position: int | None = Field(None, description="Jobs ahead of this one while it is queued (0 = next)")
    temperature_fallback: bool = Field(False, description="Re-decode windows whose text repeats or is unsure")
    condition_on_previous_text: bool = Field(False, description="Prompt each window with the text before it")
    initial_prompt: str | None = Field(None, description="Vocabulary or style hint given to every window")
    encoder_precision: str | None = Field(None, description="Requested encoder precision (fp32, int8); None = default")
    decoder_precision: str | None = Field(None, description="Requested decoder precision (fp32, int8); None = default")
    precision: dict[str, str] | None = Field(None, description="Precision actually used: {encoder, decoder}")
    error: str | None = None
    model: str | None = None
    created_at: datetime = Field(default_factory=utcnow)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    metrics: InferenceMetrics | None = None
    transcript: Transcript | None = None


class JobSummary(BaseModel):
    id: str
    filename: str
    status: JobStatus
    status_url: str
    transcript_url: str


class TranscriptResponse(Transcript):
    model: str | None
    precision: dict[str, str] | None = Field(None, description="Encoder and decoder precision used: fp32 or int8")
    metrics: InferenceMetrics | None


class HealthResponse(BaseModel):
    status: str
    model: str | None
    model_error: str | None = None
    languages: list[str]
    queue_depth: int
    active_jobs: int


# ------------------------------------------------------------------ benchmark


class BenchmarkSource(BaseModel):
    """A real recording uploaded to pick reference intervals from."""

    id: str
    filename: str
    size_bytes: int
    media_type: str
    language: str | None = None
    created_at: datetime = Field(default_factory=utcnow)
    job_id: str | None = Field(None, description="Transcription job that proposes the intervals")
    status: JobStatus = JobStatus.queued
    error: str | None = None
    duration_seconds: float | None = None
    segments: list[TranscriptSegment] | None = Field(None, description="Whisper's segments, once transcribed")


class ClipCreate(BaseModel):
    source_id: str
    start: float = Field(ge=0)
    end: float = Field(gt=0)
    reference: str = Field(min_length=1, max_length=20_000, description="The correct transcript of the interval")
    name: str | None = Field(None, max_length=200)


class ClipPatch(BaseModel):
    reference: str | None = Field(None, min_length=1, max_length=20_000)
    name: str | None = Field(None, min_length=1, max_length=200)


class ReferenceClipInfo(BaseModel):
    """An audio interval with its correct transcript: one item of the benchmark set."""

    id: str
    name: str
    reference: str
    duration_seconds: float
    source_id: str | None = None
    source_filename: str | None = None
    start: float | None = Field(None, description="Where the interval starts in the source recording")
    end: float | None = None
    created_at: datetime = Field(default_factory=utcnow)


class BenchmarkModel(BaseModel):
    id: str
    name: str
    default: bool = Field(description="The model the service transcribes with")


class BenchmarkVariant(BaseModel):
    id: str
    label: str
    temperature_fallback: bool
    condition_on_previous_text: bool


class BenchmarkCreate(BaseModel):
    models: list[str] = Field(min_length=1, max_length=10)
    variants: list[str] = Field(min_length=1, max_length=10)
    language: str = Field("auto", pattern=r"^([a-z]{2,3}|auto)$")
    initial_prompt: str | None = Field(None, max_length=1000)
    clip_ids: list[str] | None = Field(None, description="Clips to use; all when omitted")


class ClipResult(BaseModel):
    clip_id: str
    name: str
    wer: float
    errors: int
    reference_words: int
    audio_seconds: float
    inference_seconds: float
    hypothesis: str
    alignment: list[list[str]] = Field(description="[op, reference word, hypothesis word]; op: ok, sub, del, ins")


class BenchmarkResult(BaseModel):
    model: str
    model_name: str | None = None
    variant: str
    error: str | None = None
    wer: float | None = Field(None, description="All errors / all reference words, over every clip")
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0
    reference_words: int = 0
    audio_seconds: float = 0.0
    inference_seconds: float = 0.0
    real_time_factor: float | None = None
    load_seconds: float | None = None
    peak_memory_mb: float | None = None
    clips: list[ClipResult] = Field(default_factory=list)


class BenchmarkProgress(BaseModel):
    done: int = 0
    total: int = 0
    current: str | None = None


class BenchmarkRun(BaseModel):
    id: str
    status: JobStatus = JobStatus.queued
    models: list[str]
    variants: list[str]
    language: str
    initial_prompt: str | None = None
    clip_ids: list[str]
    created_at: datetime = Field(default_factory=utcnow)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None
    progress: BenchmarkProgress = Field(default_factory=BenchmarkProgress)
    results: list[BenchmarkResult] = Field(default_factory=list)
