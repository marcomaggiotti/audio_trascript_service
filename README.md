# Audio Transcript Service

Send an audio file, get its text back: **Whisper running on ONNX Runtime**, with timestamped
segments, plain text, SRT or WebVTT output. A web page at `/` has two tabs: **Transcribe**, and
**Benchmark**, which measures the word error rate of models and decoding options on reference
intervals you pick from your own recordings.

The speech-recognition code is taken from
[voice_embedding_service](https://github.com/marcomaggiotti/voice_embedding_service) (Whisper ONNX
export, NumPy log-mel features, greedy decoding with Whisper's timestamp rules, audio decoding).
This service keeps only that part. It has no search or embeddings.

```
 POST /transcribe ─────────────► queue ─► ffmpeg decode → 16 kHz mono
 POST /jobs  (202 + job id)              log-mel spectrogram (NumPy)
                                         Whisper encoder / decoder with KV cache (ONNX Runtime)
 GET /jobs/{id}/transcript ◄──────────── timestamped segments → json | text | srt | vtt
```

## What's in it

| Area | Implementation |
|---|---|
| Speech recognition | Whisper exported to ONNX (fp32 encoder, INT8 KV-cache decoder: an INT8 encoder garbles real speech, see `scripts/export_whisper_onnx.py`). Whisper's timestamp rules, 30 s windows with seek, language auto-detect, temperature fallback, previous-text and initial prompts (`app/asr.py`) |
| Features | Whisper log-mel spectrogram in pure NumPy, no torch or librosa at runtime (`app/features.py`) |
| Audio | Magic-byte format check, ffmpeg decoding to 16 kHz mono with a pure-Python WAV fallback (`app/audio.py`) |
| Jobs | In-process asyncio queue with bounded concurrency. Synchronous requests use the same queue (`app/jobs.py`) |
| Output | JSON with segments and per-segment confidence, plain text, SRT, WebVTT (`app/formats.py`) |
| Metrics | Per job: decode time, inference time, real-time factor, peak memory |
| Benchmark | Reference clips cut from real recordings, word error rate per model and decoding variant, word-level diffs (`app/benchmark.py`, `app/wer.py`, `scripts/wer.py`) |
| Web page | `GET /`: Transcribe and Benchmark tabs (`app/static/index.html`) |

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/transcribe?format=json\|text\|srt\|vtt` | Multipart `file` (WAV, MP3, OGG/Opus, M4A, FLAC) and optional `language` (`en`, `de`, `it`, … or `auto`), `temperature_fallback`, `condition_on_previous_text` (`true`/`false`) and `initial_prompt`. Waits and returns the transcript |
| `POST` | `/jobs` | Same upload, transcribed in the background. Returns `202` with the job id, `status_url` and `transcript_url` |
| `GET` | `/jobs` | Recent jobs, newest first |
| `GET` | `/jobs/{id}` | Status, `stage` (decoding, transcribing), `progress` (0–1, updated after each 30 s window), `queue_position` while queued, model, timings and metrics, or the error if it failed |
| `GET` | `/jobs/{id}/transcript?format=` | The transcript. `409` until it's ready, `422` if the job failed |
| `DELETE` | `/jobs/{id}` | Forget a job (and its upload if it's still queued) |
| `GET` | `/config` | How audio becomes text here: decoding, windows, log-mel features (fixed by Whisper) and the model, its encoder/decoder precision (read from the ONNX graphs) and decoding defaults |
| `GET` | `/languages` | Languages the loaded model supports, and the default |
| `GET` | `/health` | Model status, queue depth, active jobs |
| `GET` | `/healthz` | Liveness: `503` if the speech model didn't load |
| | `/benchmark/…`, `/benchmarks` | The benchmark: see [below](#benchmark-word-error-rate-on-your-own-audio) |

```bash
# Plain text, waiting for the result
curl -F file=@episode.mp3 "localhost:8000/transcribe?format=text"

# Subtitles in Italian
curl -F file=@clip.m4a -F language=it "localhost:8000/transcribe?format=srt" -o clip.srt

# Long file: queue it, poll, fetch
curl -F file=@talk.mp3 localhost:8000/jobs               # {"id": "…", "status": "queued", …}
curl localhost:8000/jobs/<id>                            # "status": "completed"
curl localhost:8000/jobs/<id>/transcript
```

The JSON transcript looks like this:

```json
{
  "language": "en",
  "duration_seconds": 15.0,
  "text": "Welcome to the show. Today we talk about ONNX Runtime.",
  "segments": [
    {"index": 0, "start": 0.0, "end": 4.2, "text": "Welcome to the show.",
     "avg_logprob": -0.21, "no_speech_prob": 0.01}
  ],
  "model": "openai/whisper-tiny",
  "metrics": {"audio_seconds": 15.0, "decode_seconds": 0.05, "inference_seconds": 2.1,
              "real_time_factor": 0.14, "peak_memory_mb": 310.0}
}
```

`avg_logprob` is the mean log-probability of the segment's tokens: below about −0.7 the text is often
wrong. `no_speech_prob` is Whisper's estimate that the segment's 30 s window is silence.

## Decoding options

Each 30 s window is decoded greedily (the most likely token each step). Three options, all on by
default except the prompt, follow openai/whisper:

- **Temperature fallback** (`TEMPERATURE_FALLBACK`, form field `temperature_fallback`): a window whose
  text repeats itself (zlib compression ratio above 2.4) or whose tokens are unsure (mean log-probability
  below −1) is decoded again, sampling at temperatures 0.2, 0.4 … 1.0, until it passes. A window Whisper
  thinks is silence (no-speech probability above 0.6, text unsure) gives no text at all. This fixes most
  hallucinations and repetition loops, at the cost of extra decoding on bad windows only.
- **Previous-text prompt** (`CONDITION_ON_PREVIOUS_TEXT`, `condition_on_previous_text`): each window is
  prompted with the text decoded before it (up to 223 tokens), which keeps names and spelling consistent.
  It's dropped after a window that needed a high temperature, so one bad window doesn't spread.
- **Initial prompt** (`INITIAL_PROMPT`, `initial_prompt`): words and spellings the audio contains, such as
  names and product terms, given to every window.

Sampling uses a fixed seed, so the same audio gives the same transcript. Whether these options help
depends on the model and the audio: measure it with the benchmark.

## Benchmark: word error rate on your own audio

Speed tells you little about which model to use; accuracy on *your* audio does. The **Benchmark** tab
(`GET /`) scores models and decoding variants by word error rate (WER):

1. **Pick reference intervals.** Upload a real recording. Whisper transcribes it and shows its
   segments on a timeline, colour-coded by confidence. Select a segment, or shift-click to span several,
   adjust start and end to the tenth of a second (or set them from the player's position), play the
   interval, and type what is *really* said. Save it as a reference clip. Choose intervals that are
   representative: clear, complete sentences, from different speakers and conditions. About 20–30 clips
   give a stable score. You can also import clips you cut yourself, with their transcripts.
2. **Reference clips** lists the set. Transcripts can be edited in place; unticked clips are skipped.
3. **Run** with any models and decoding variants (greedy, temperature fallback, fallback + previous
   text), a language and an optional initial prompt. Runs go through the transcription queue, so they
   never run at the same time as a job.
4. **Results** shows WER, substitutions, deletions and insertions, real-time factor, peak memory and
   load time for each combination, with the lowest WER highlighted. Click a row for every clip's
   word-level differences against its reference.

WER = (wrong + missing + extra words) / words in the references, after lower-casing and removing
punctuation and bracketed notes like `[music]`.

**Models to compare** come from `MODELS_DIR` (default `./models`): every subdirectory holding an export.
Export the candidates next to each other:

```bash
python scripts/export_whisper_onnx.py --model openai/whisper-tiny --output models/whisper-tiny
python scripts/export_whisper_onnx.py --model openai/whisper-base --output models/whisper-base
```

The service's own model stays loaded; another model is loaded for its part of a run and freed
afterwards, so a run needs room for both (whisper-base with tiny: about +150 MB).

**On disk** (`BENCHMARK_DIR`, default `./data/benchmark`): recordings in `sources/`, reference clips in
`clips/` as `<id>.wav` + `<id>.txt` (the correct transcript) + `<id>.json`, and finished runs in `runs/`.

**From the command line**, the same scoring works on any folder of audio files with same-name `.txt`
references, including the Benchmark tab's `clips/` folder:

```bash
python scripts/wer.py data/benchmark/clips --model-dir models/whisper-tiny --model-dir models/whisper-base \
    --language it --details
```

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/benchmark/sources` | Upload a recording (`file`, `language`); Whisper proposes its segments |
| `GET` | `/benchmark/sources`, `/benchmark/sources/{id}` | Recordings; one with its proposed segments |
| `GET` | `/benchmark/sources/{id}/audio` | The recording |
| `POST` | `/benchmark/sources/{id}/transcribe?language=` | Propose segments again, e.g. in another language |
| `DELETE` | `/benchmark/sources/{id}` | Delete a recording (its clips stay) |
| `POST` | `/benchmark/clips` | `{"source_id", "start", "end", "reference", "name"}`: cut an interval into a reference clip |
| `POST` | `/benchmark/clips/upload` | Multipart `file`, `reference`, `name`: a clip you cut yourself |
| `GET` / `PATCH` / `DELETE` | `/benchmark/clips[/{id}]` | List, edit the transcript or name, delete |
| `GET` | `/benchmark/clips/{id}/audio` | The clip as 16 kHz WAV |
| `GET` | `/benchmark/models`, `/benchmark/variants` | What a run can compare |
| `POST` | `/benchmarks` | `{"models", "variants", "language", "initial_prompt", "clip_ids"}`: queue a run (`202`) |
| `GET` | `/benchmarks`, `/benchmarks/{id}` | Runs with progress and results; one run with per-clip diffs |
| `DELETE` | `/benchmarks/{id}` | Delete a finished run |

When `API_TOKEN` is set, `/transcribe`, `/jobs`, `/languages`, `/benchmark/…` and `/benchmarks` need
`X-API-Key: <token>` or `Authorization: Bearer <token>`. The web page asks for the token once (top
right). The page itself, `/health`, `/healthz` and the API docs (`/docs`) stay public.

### Encoder and decoder precision

The image keeps both precisions of each part of the model: the encoder in fp32 (the default) and int8, the
decoder in int8 (the default) and fp32. `POST /jobs` and `POST /transcribe` take `encoder=fp32|int8` and
`decoder=fp32|int8`; the job and its transcript record the pair used (`precision`). The default pair stays
loaded; a job that asks for another part swaps it in (the default one is released first, the requested one
loaded in under a second for tiny) and the default is loaded back afterwards, so two versions of a part are
never in memory together. Measured on whisper-tiny, one thread, low-memory mode, a 6 s voice note:

| Encoder / decoder | Peak RSS | Time | Note |
|---|---|---|---|
| fp32 / int8 (default) | 290 MB | 0.6 s | |
| fp32 / fp32 | 312 MB | 1.5 s | most accurate (WER 5.0% vs 5.7% on the test set) |
| int8 / int8 | 209 MB | 0.9 s | garbles real speech |
| int8 / fp32 | 288 MB | 1.3 s | garbles real speech |

`GET /config` lists what the export has (`model.available_precisions`). `--no-variants` on the export keeps
only the default pair (about 120 MB smaller image; jobs can't choose).

## Run locally

```bash
pip install -r requirements-dev.txt
# Once, to export the model. Needs torch (CPU build is enough):
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install -r requirements-export.txt
python scripts/export_whisper_onnx.py --model openai/whisper-tiny --output models/whisper-tiny

WHISPER_MODEL_DIR=models/whisper-tiny python main.py
open http://localhost:8000          # web page; API docs at /docs
```

`ffmpeg` must be on the PATH for MP3, OGG, M4A and FLAC. WAV falls back to a pure-Python decoder.
Without a model directory the service still starts in `degraded` mode: `/health` says why, and
transcription requests get `503`.

Any Whisper checkpoint on Hugging Face works: `openai/whisper-base`, `openai/whisper-small`, or an
English-only `openai/whisper-tiny.en`. Larger models are more accurate but slower and use more RAM.

Tests use a fake Whisper, so they need no model download:

```bash
pytest
```

## Configuration

| Variable | Default | Notes |
|---|---|---|
| `WHISPER_MODEL_DIR` | `./models/whisper-tiny` | Output of `scripts/export_whisper_onnx.py` |
| `DEFAULT_LANGUAGE` | `auto` | Used when a request doesn't specify one; `auto` detects it |
| `MAX_NEW_TOKENS` | `224` | Decoder tokens per 30 s window |
| `ONNX_THREADS` | `0` (auto) | ONNX Runtime threads; set to the instance's CPU count |
| `WHISPER_LOW_MEMORY` | `false` | About 25 MB less RAM, about 1.5x slower transcription |
| `TEMPERATURE_FALLBACK` / `CONDITION_ON_PREVIOUS_TEXT` | `true` / `true` | Decoding defaults; requests can override them |
| `INITIAL_PROMPT` | – | Default initial prompt (names, terms) |
| `MODELS_DIR` | `./models` | Exports the benchmark can compare |
| `BENCHMARK_DIR` | `./data/benchmark` | Recordings, reference clips and runs |
| `MAX_CLIP_SECONDS` | `300` | Longest reference clip |
| `API_TOKEN` | – | Requires a token for the transcription endpoints |
| `MAX_UPLOAD_MB` / `MAX_AUDIO_SECONDS` | `50` / `1800` | Upload limits; decoding stops just after the duration limit |
| `WORKER_CONCURRENCY` / `MAX_QUEUE_SIZE` | `1` / `20` | Parallel transcriptions; uploads get `503` when the queue is full |
| `MAX_FINISHED_JOBS` | `200` | Finished jobs kept for `GET /jobs/{id}`; older ones are forgotten |
| `UPLOAD_DIR` | `./data/uploads` | Uploads wait here until decoded, then are deleted |

## Docker and Render

```bash
docker build -t audio-transcript --build-arg WHISPER_MODEL=openai/whisper-tiny .
docker run -p 10000:10000 -e API_TOKEN=secret audio-transcript
```

The Dockerfile exports Whisper to ONNX in a throwaway build stage (torch never reaches the runtime
image), so the container needs no downloads at boot. To compare other models in the Benchmark tab,
export them into the image too: `--build-arg BENCHMARK_MODELS="openai/whisper-base openai/whisper-small"`.
Mount a volume at `/data` to keep the benchmark's recordings, clips and runs.

`render.yaml` is a Render Blueprint for the same image: **New → Blueprint**, pick this repository,
then copy the generated `API_TOKEN` from the service's **Environment** tab. A 1 GB disk at `/data` keeps
the benchmark set across deploys. To change model size, edit `WHISPER_MODEL` in `render.yaml` and push;
it's passed to the Docker build, like `BENCHMARK_MODELS`. On the 512 MB starter plan, comparing a second
model next to whisper-tiny may run out of memory: use `standard` for benchmarks with whisper-base or larger.

## Speed benchmark

```bash
python scripts/benchmark.py sample.mp3 --model-dir models/whisper-tiny --runs 3
```

It prints inference latency, real-time factor and peak memory on this machine.

## Limitations

- Job state is in memory. A restart forgets queued and finished jobs. Run one process (the default
  `python main.py`). For several instances, move the queue to Redis (RQ/arq);
  `TranscriptionService.process()` doesn't need to change.
- No beam search: each attempt is greedy or sampled, as in openai/whisper's default settings.
- A benchmark run interrupted by a restart is marked failed; start it again.
