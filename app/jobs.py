"""Background transcription jobs: decode -> Whisper -> timestamped transcript.

Jobs run on an in-process asyncio queue; the CPU-bound work runs in worker threads and
`concurrency` bounds how many inferences run at once (Whisper's memory peak is per job).
Synchronous requests and benchmark runs go through the same queue, so they are bounded the same way.
Job state lives in memory: a restart forgets it, and the uploads waiting in the queue are removed.
"""

import asyncio
import functools
import logging
import time
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path

from app.asr import DecodingOptions, Transcriber
from app.audio import InvalidAudioError, decode_audio, duration_seconds
from app.metrics import peak_memory_mb
from app.schemas import InferenceMetrics, Job, JobStatus, Transcript, TranscriptSegment, utcnow

logger = logging.getLogger(__name__)


class QueueFullError(RuntimeError):
    pass


class JobNotFound(KeyError):
    pass


class TranscriptionService:
    def __init__(self, transcriber: Transcriber | None, upload_dir: str, transcriber_error: str | None = None,
                 max_audio_seconds: int = 1800, concurrency: int = 1, max_queue_size: int = 20,
                 max_finished_jobs: int = 200):
        self.transcriber = transcriber
        self.transcriber_error = transcriber_error
        self.upload_dir = Path(upload_dir)
        self.max_audio_seconds = max_audio_seconds
        self.concurrency = concurrency
        self.max_finished_jobs = max_finished_jobs
        self.queue: asyncio.Queue[str] = asyncio.Queue(maxsize=max_queue_size)
        self.jobs: OrderedDict[str, Job] = OrderedDict()
        self._finished: dict[str, asyncio.Event] = {}
        self._tasks: dict[str, Callable[[], None]] = {}  # queued work that isn't a transcription job
        self._workers: list[asyncio.Task] = []
        self.active = 0  # jobs currently being transcribed

    @property
    def languages(self) -> list[str]:
        return list(getattr(self.transcriber, "supported_languages", None) or [])

    def upload_path(self, job_id: str) -> Path:
        return self.upload_dir / job_id

    # ------------------------------------------------------------- lifecycle

    async def start(self) -> None:
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        # Jobs are in memory only, so uploads left by a previous process can never be picked up.
        for leftover in self.upload_dir.iterdir():
            if leftover.is_file():
                leftover.unlink(missing_ok=True)
        self._workers = [asyncio.create_task(self._worker(i)) for i in range(self.concurrency)]

    async def stop(self) -> None:
        for task in self._workers:
            task.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)

    # ------------------------------------------------------------------ jobs

    def submit(self, job: Job) -> None:
        """Queue a job whose audio is already at upload_path(job.id).

        On a full queue the upload is removed and QueueFullError is raised.
        """
        self.jobs[job.id] = job
        self._finished[job.id] = asyncio.Event()
        try:
            self.queue.put_nowait(job.id)
        except asyncio.QueueFull as e:
            self.delete(job.id)
            raise QueueFullError("Transcription queue is full, try again later") from e

    def get(self, job_id: str) -> Job:
        try:
            job = self.jobs[job_id]
        except KeyError:
            raise JobNotFound(job_id) from None
        if job.status == JobStatus.queued:  # jobs are kept in submission order
            ahead = 0
            for other in self.jobs.values():
                if other.id == job_id:
                    break
                ahead += other.status == JobStatus.queued
            job.queue_position = ahead
        else:
            job.queue_position = None
        return job

    def list(self) -> list[Job]:
        return list(reversed(self.jobs.values()))

    def delete(self, job_id: str) -> None:
        """Forget a job; a queued one is skipped, a running one finishes and is discarded."""
        self.get(job_id)
        del self.jobs[job_id]
        self._finished.pop(job_id, None)
        self.upload_path(job_id).unlink(missing_ok=True)

    def submit_task(self, task_id: str, run: Callable[[], None]) -> None:
        """Queue other inference work (a benchmark run) behind the jobs, on the same workers."""
        self._tasks[task_id] = run
        try:
            self.queue.put_nowait(task_id)
        except asyncio.QueueFull as e:
            del self._tasks[task_id]
            raise QueueFullError("Transcription queue is full, try again later") from e

    async def wait(self, job_id: str) -> Job:
        """Wait until the job is completed or failed and return it."""
        job = self.get(job_id)
        event = self._finished.get(job_id)
        if event is not None:
            await event.wait()
        return job

    async def drain(self) -> None:
        """Wait until every queued job has finished (used by tests and benchmarks)."""
        await self.queue.join()

    async def _worker(self, worker_id: int) -> None:
        while True:
            job_id = await self.queue.get()
            task = self._tasks.pop(job_id, None)
            try:
                self.active += 1
                try:
                    await asyncio.to_thread(task or functools.partial(self.process, job_id))
                finally:
                    self.active -= 1
            except Exception:  # process() records failures itself; never kill the worker
                logger.exception("Worker %d crashed on %s", worker_id, job_id)
            finally:
                if (event := self._finished.pop(job_id, None)) is not None:
                    event.set()
                self._prune()
                self.queue.task_done()

    def _prune(self) -> None:
        finished = [j.id for j in self.jobs.values() if j.status in (JobStatus.completed, JobStatus.failed)]
        for job_id in finished[: max(len(finished) - self.max_finished_jobs, 0)]:
            self.jobs.pop(job_id, None)

    # ------------------------------------------------------------------- job

    def process(self, job_id: str) -> None:
        job = self.jobs.get(job_id)
        if job is None:
            return  # deleted while queued
        job.status, job.started_at, job.stage = JobStatus.processing, utcnow(), "decoding"
        path = self.upload_path(job_id)
        try:
            if self.transcriber is None:
                raise RuntimeError(f"Speech model unavailable: {self.transcriber_error or 'not loaded'}")

            t0 = time.perf_counter()
            audio = decode_audio(path, self.max_audio_seconds, as_int16=True)
            path.unlink(missing_ok=True)  # decoded: the upload isn't needed any more
            seconds = duration_seconds(audio)
            if seconds > self.max_audio_seconds:
                raise InvalidAudioError(f"Audio is longer than the {self.max_audio_seconds}s limit")
            t1 = time.perf_counter()
            job.stage = "transcribing"

            def progress(share: float) -> None:
                job.progress = round(share, 3)

            options = DecodingOptions(temperature_fallback=job.temperature_fallback,
                                      condition_on_previous_text=job.condition_on_previous_text,
                                      initial_prompt=job.initial_prompt)
            switch = getattr(self.transcriber, "precision_for", None)  # absent on test fakes
            if switch is None:
                job.precision = getattr(self.transcriber, "precision", None)
                result = self.transcriber.transcribe(audio, language=job.language, on_progress=progress,
                                                     options=options)
            else:
                with switch(job.encoder_precision, job.decoder_precision) as used:
                    job.precision = used
                    result = self.transcriber.transcribe(audio, language=job.language, on_progress=progress,
                                                         options=options)
            t2 = time.perf_counter()
        except Exception as e:
            logger.exception("Transcription failed for %s", job_id)
            job.error = str(e) or e.__class__.__name__
            job.status, job.finished_at, job.stage = JobStatus.failed, utcnow(), None
            path.unlink(missing_ok=True)
            return

        job.model = self.transcriber.name
        job.transcript = Transcript(
            language=result.language or job.language,
            duration_seconds=round(seconds, 2),
            text=result.text,
            segments=[TranscriptSegment(index=i, start=s.start, end=s.end, text=s.text,
                                        avg_logprob=getattr(s, "avg_logprob", None),
                                        no_speech_prob=getattr(s, "no_speech_prob", None))
                      for i, s in enumerate(result.segments)],
        )
        job.metrics = InferenceMetrics(
            audio_seconds=round(seconds, 2),
            decode_seconds=round(t1 - t0, 3),
            inference_seconds=round(t2 - t1, 3),
            real_time_factor=round((t2 - t1) / seconds, 3) if seconds else 0.0,
            peak_memory_mb=peak_memory_mb(),
        )
        # Status last, so a reader that sees "completed" also sees the transcript.
        job.progress, job.stage = 1.0, None
        job.status, job.finished_at = JobStatus.completed, utcnow()
        logger.info("Transcribed %s: %.1fs audio in %.2fs", job_id, seconds, t2 - t1)
