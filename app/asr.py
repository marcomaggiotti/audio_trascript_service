"""Speech recognition with a Whisper model exported to ONNX, run through ONNX Runtime.

Expects the directory layout produced by scripts/export_whisper_onnx.py (optimum export, then INT8):

    encoder_model.onnx          input_features -> last_hidden_state
    decoder_model_merged.onnx   input_ids + encoder_hidden_states + past_key_values.* + use_cache_branch
                                -> logits + present.* (KV cache: one new token per decoder call)
      or decoder_model.onnx     input_ids + encoder_hidden_states -> logits (legacy, no KV cache:
                                every step re-runs the whole prefix and returns [1, n, vocab] logits)
    tokenizer.json
    generation_config.json    (optional, provides suppress/lang token ids)
    preprocessor_config.json  (optional, provides n_mels)

Decoding is greedy with Whisper's timestamp rules, so every transcript comes back as
segments with start/end times. Long audio is processed in 30 s windows that advance
to the last complete timestamp, the same "seek" strategy openai/whisper uses.
DecodingOptions adds openai/whisper's temperature fallback (re-decode a window whose text
repeats or is unsure), prompting with the previous text, and an initial prompt.
"""

import json
import logging
import os
import zlib
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from app.features import (
    N_FRAMES,
    N_SAMPLES,
    SAMPLE_RATE,
    log_mel_spectrogram,
    pad_or_trim,
)

logger = logging.getLogger(__name__)

TIME_PRECISION = 0.02  # seconds per timestamp token
MAX_INITIAL_TIMESTAMP = 1.0  # first segment must start within 1 s of the window
N_TEXT_CTX = 448  # decoder positions: prompt + sampled tokens must fit
MAX_PROMPT_TOKENS = N_TEXT_CTX // 2 - 1  # previous-text tokens kept in the prompt (as openai/whisper)

# Temperature fallback, with openai/whisper's thresholds: a window is decoded again, sampling at the
# next temperature, when its text compresses too well (repetition loop) or its tokens are unsure.
FALLBACK_TEMPERATURES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
COMPRESSION_RATIO_THRESHOLD = 2.4
LOGPROB_THRESHOLD = -1.0
# With fallback on, a window Whisper thinks is silent (and whose text is unsure) produces no segments.
NO_SPEECH_THRESHOLD = 0.6


@dataclass(frozen=True)
class DecodingOptions:
    """How each 30 s window is decoded. The defaults are plain greedy decoding."""

    temperature_fallback: bool = False  # re-decode a window at rising temperatures when its text looks wrong
    condition_on_previous_text: bool = False  # prompt each window with the text decoded before it
    initial_prompt: str | None = None  # vocabulary or style hint, prompted on every window
    seed: int = 0  # for the sampled (temperature > 0) attempts, so runs are repeatable


@dataclass
class Segment:
    start: float
    end: float
    text: str
    # How sure Whisper was (None from transcribers that don't measure it):
    avg_logprob: float | None = None  # mean log-probability of the segment's text tokens
    no_speech_prob: float | None = None  # Whisper's estimate that its 30 s window is silence
    tokens: list[int] = field(default_factory=list, repr=False, compare=False)  # text token ids


@dataclass
class TranscriptionResult:
    language: str | None
    segments: list[Segment] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(s.text for s in self.segments).strip()


@dataclass
class WindowResult:
    segments: list[Segment]  # relative to the window
    consumed: float  # seconds of the window covered by the segments
    avg_logprob: float
    compression_ratio: float
    temperature: float


class Transcriber(Protocol):
    name: str

    def transcribe(self, audio: np.ndarray, language: str | None = None,
                   on_progress: Callable[[float], None] | None = None,
                   options: DecodingOptions | None = None) -> TranscriptionResult: ...


class ModelLoadError(RuntimeError):
    pass


PRECISIONS = ("fp32", "int8")


def _release_memory() -> None:
    """Hand freed memory back to the OS (glibc keeps it otherwise, so RSS would stay at the swap's peak)."""
    import ctypes
    import gc

    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):  # not glibc (macOS, musl): nothing to do
        pass


def onnx_precision(path: Path) -> str:
    """'int8' if the graph was dynamically quantized (scripts/export_whisper_onnx.py), else 'fp32'.

    Op types are stored as plain strings in the protobuf, so this needs no onnx package and no full parse.
    """
    import mmap

    with path.open("rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as data:
        return "int8" if data.find(b"DynamicQuantizeLinear") >= 0 or data.find(b"MatMulInteger") >= 0 else "fp32"


@dataclass
class WhisperTokens:
    """Special token ids, resolved from the tokenizer and generation config."""

    eot: int
    sot: int
    transcribe: int | None
    no_timestamps: int
    timestamp_begin: int
    lang_to_id: dict[str, int]
    suppress: list[int]
    no_speech: int | None = None  # <|nospeech|> (<|nocaptions|> in older checkpoints)
    prev: int | None = None  # <|startofprev|>: starts the previous-text prompt

    @property
    def multilingual(self) -> bool:
        return bool(self.lang_to_id)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def _resolve_tokens(tokenizer, generation_config: dict) -> WhisperTokens:
    def tid(token: str) -> int | None:
        return tokenizer.token_to_id(token)

    eot, sot = tid("<|endoftext|>"), tid("<|startoftranscript|>")
    no_ts = generation_config.get("no_timestamps_token_id") or tid("<|notimestamps|>")
    if eot is None or sot is None or no_ts is None:
        raise ModelLoadError("tokenizer.json is missing Whisper special tokens")

    lang_to_id = {
        tok.strip("<|>"): idx for tok, idx in (generation_config.get("lang_to_id") or {}).items()
    }
    if not lang_to_id and generation_config.get("is_multilingual", True):
        for code in ("en", "de", "fr", "it", "es", "pt", "nl", "ja", "zh", "ru"):
            if (idx := tid(f"<|{code}|>")) is not None:
                lang_to_id[code] = idx

    return WhisperTokens(
        eot=eot,
        sot=sot,
        transcribe=tid("<|transcribe|>"),
        no_timestamps=no_ts,
        timestamp_begin=no_ts + 1,
        lang_to_id=lang_to_id,
        suppress=[t for t in generation_config.get("suppress_tokens", []) if t >= 0],
        no_speech=tid("<|nospeech|>") if tid("<|nospeech|>") is not None else tid("<|nocaptions|>"),
        prev=tid("<|startofprev|>"),
    )


class OnnxWhisperTranscriber:
    """Greedy Whisper decoding on ONNX Runtime (CPU by default)."""

    def __init__(self, encoder, decoder, tokenizer, tokens: WhisperTokens, n_mels: int = 80,
                 max_new_tokens: int = 224, name: str = "whisper-onnx"):
        self.encoder = encoder
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.tokens = tokens
        self.n_mels = n_mels
        self.max_new_tokens = max_new_tokens
        self.name = name
        # Precision of each part, and the files of every precision exported (from_dir fills these in).
        self.precision: dict[str, str] | None = None
        self.variants: dict[str, dict[str, Path]] = {}
        self._load_session: Callable[[Path], object] | None = None
        self._bind(encoder, decoder)

    def _bind(self, encoder, decoder) -> None:
        """Use these sessions, and read the input/output layout the decoding loop needs from them."""
        self.encoder, self.decoder = encoder, decoder
        self._encoder_input = encoder.get_inputs()[0].name
        decoder_inputs = {i.name: i for i in decoder.get_inputs()}
        if {"input_ids", "encoder_hidden_states"} - decoder_inputs.keys():
            raise ModelLoadError(
                f"Unsupported decoder inputs {sorted(decoder_inputs)}; "
                "export with scripts/export_whisper_onnx.py (decoder_model_merged.onnx or decoder_model.onnx)"
            )
        # optimum's merged decoder: an If node picks the first-call branch (no past) or the cached one.
        self._kv_cache = "use_cache_branch" in decoder_inputs
        self._decoder_outputs = [o.name for o in decoder.get_outputs()] if self._kv_cache else ["logits"]
        # past_key_values.<i>.{decoder,encoder}.{key,value}: [batch, heads, seq, head_dim], seq = 0 at start
        self._empty_past = {name: (1, inp.shape[1], 0, inp.shape[3]) for name, inp in decoder_inputs.items()
                            if name.startswith("past_key_values.")}

    @property
    def supported_languages(self) -> list[str]:
        """Language codes the model can transcribe (English-only models: just en)."""
        return list(self.tokens.lang_to_id) if self.tokens.multilingual else ["en"]

    @classmethod
    def from_dir(cls, model_dir: str, max_new_tokens: int = 224, threads: int = 0,
                 low_memory: bool = False) -> "OnnxWhisperTranscriber":
        import onnxruntime as ort
        from tokenizers import Tokenizer

        path = Path(model_dir)
        decoder_file = next((n for n in ("decoder_model_merged.onnx", "decoder_model.onnx") if (path / n).exists()),
                            "decoder_model_merged.onnx")
        for name in ("encoder_model.onnx", decoder_file, "tokenizer.json"):
            if not (path / name).exists():
                raise ModelLoadError(
                    f"{path / name} not found. Export a model first: python scripts/export_whisper_onnx.py"
                )

        ort.disable_telemetry_events()
        options = ort.SessionOptions()
        if threads:
            options.intra_op_num_threads = threads
        # No arena: it keeps the encoder's peak (attention, mel) allocated after every window. With the KV
        # cache each decoder call is tiny, so plain malloc costs nothing measurable.
        options.enable_cpu_mem_arena = False
        options.enable_mem_pattern = False
        if low_memory:  # ~-25 MB RSS (no pre-packed weight copies), decoder steps ~2x slower
            options.add_session_config_entry("session.disable_prepacking", "1")
        providers = ["CPUExecutionProvider"]
        encoder = ort.InferenceSession(str(path / "encoder_model.onnx"), options, providers=providers)
        decoder = ort.InferenceSession(str(path / decoder_file), options, providers=providers)
        tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))

        tokens = _resolve_tokens(tokenizer, _load_json(path / "generation_config.json"))
        n_mels = _load_json(path / "preprocessor_config.json").get("feature_size", 80)
        model_id = path / "model_id.txt"  # written by scripts/export_whisper_onnx.py
        name = model_id.read_text().strip() if model_id.exists() else os.path.basename(path.resolve())
        transcriber = cls(encoder, decoder, tokenizer, tokens, n_mels=n_mels, max_new_tokens=max_new_tokens, name=name)
        transcriber.precision = {"encoder": onnx_precision(path / "encoder_model.onnx"),
                                 "decoder": onnx_precision(path / decoder_file)}
        # The other precision of each part, if the export kept it (encoder_model.int8.onnx, ...).
        for part, default_file in (("encoder", "encoder_model.onnx"), ("decoder", decoder_file)):
            stem = default_file.removesuffix(".onnx")
            files = {transcriber.precision[part]: path / default_file}
            for p in PRECISIONS:
                if p not in files and (path / f"{stem}.{p}.onnx").exists():
                    files[p] = path / f"{stem}.{p}.onnx"
            transcriber.variants[part] = files
        transcriber._load_session = lambda file: ort.InferenceSession(str(file), options, providers=providers)
        return transcriber

    @property
    def available_precisions(self) -> dict[str, list[str]]:
        """Precisions a job can ask for, per part ({} if unknown: then only the default)."""
        return {part: sorted(files) for part, files in self.variants.items()}

    @contextmanager
    def precision_for(self, encoder: str | None = None, decoder: str | None = None):
        """Transcribe with these precisions inside the block (None = the default), then go back.

        A part in another precision is swapped in: the default one is released first, the requested one
        loaded (1-2 s for tiny), and after the block the default is loaded again. So two versions of a part
        are never in memory together (the all-fp32 pair alone peaks ~320 MB on tiny). Jobs run one at a
        time (WORKER_CONCURRENCY=1), so swapping is safe. Yields the precision actually used per part.
        """
        default = dict(self.precision or {})
        wanted = {"encoder": encoder or default.get("encoder"), "decoder": decoder or default.get("decoder")}
        for part, p in wanted.items():
            if p != default.get(part) and p not in self.variants.get(part, {}):
                raise ValueError(f"{part} precision {p} isn't available "
                                 f"(this export has {', '.join(self.variants.get(part) or [default.get(part) or '?'])})")
        swapped = [part for part in ("encoder", "decoder") if wanted[part] != default.get(part)]
        try:
            self._swap({part: wanted[part] for part in swapped})
            yield wanted
        finally:
            if swapped:
                self._swap({part: default[part] for part in swapped})
                _release_memory()  # the variant's weights are freed by the swap: return them too

    def _swap(self, precisions: dict[str, str]) -> None:
        """Replace these parts' sessions with the given precision, releasing the old session first."""
        if not precisions:
            return
        sessions = {"encoder": self.encoder, "decoder": self.decoder}
        for part in precisions:
            sessions[part] = None
            setattr(self, part, None)  # drop the last reference before loading the replacement
        _release_memory()
        for part, p in precisions.items():
            sessions[part] = self._load_session(self.variants[part][p])
        self._bind(sessions["encoder"], sessions["decoder"])


    # ------------------------------------------------------------------ public

    def transcribe(self, audio: np.ndarray, language: str | None = None,
                   on_progress: Callable[[float], None] | None = None,
                   options: DecodingOptions | None = None) -> TranscriptionResult:
        """`on_progress` gets the share of the audio transcribed (0-1) after each 30 s window."""
        options = options or DecodingOptions()
        rng = np.random.default_rng(options.seed)
        initial = self._encode_prompt(options.initial_prompt)
        previous: list[int] = []  # text tokens of the windows decoded so far, for the next prompt
        duration = audio.shape[0] / SAMPLE_RATE
        segments: list[Segment] = []
        detected: str | None = None
        seek = 0  # in samples

        while seek < audio.shape[0]:
            window = audio[seek : seek + N_SAMPLES]
            if window.dtype == np.int16:  # 16-bit input: convert one window at a time (half the RAM)
                window = window.astype(np.float32) / 32768.0
            window_seconds = window.shape[0] / SAMPLE_RATE
            if window_seconds < 0.1:  # trailing sliver, nothing useful to decode
                break
            offset = seek / SAMPLE_RATE

            mel = log_mel_spectrogram(pad_or_trim(window), self.n_mels)[:, :N_FRAMES]
            encoded = self.encoder.run(None, {self._encoder_input: mel[None]})[0]

            # One decoder step after <|startoftranscript|> gives both the language and the no-speech odds.
            sot_logits = self._next_logits([self.tokens.sot], encoded)
            lang = language
            if self.tokens.multilingual:
                if lang in (None, "", "auto"):
                    lang = detected or self._language_from(sot_logits)
                detected = detected or lang
            no_speech = self._no_speech_prob(sot_logits)

            result = self._decode_with_fallback(encoded, lang, window_seconds, initial + previous,
                                                no_speech, options, rng)
            silent = (options.temperature_fallback and no_speech is not None and no_speech > NO_SPEECH_THRESHOLD
                      and result.avg_logprob < LOGPROB_THRESHOLD)
            if silent:  # Whisper's no-speech skip: drop the window's text, move past the whole window
                result.consumed = window_seconds
                previous = []
            else:
                for seg in result.segments:
                    start = min(offset + seg.start, duration)
                    end = min(offset + seg.end, duration)
                    if seg.text and end > start:
                        segments.append(Segment(round(start, 2), round(end, 2), seg.text,
                                                seg.avg_logprob, no_speech, seg.tokens))
                # Text sampled at a high temperature is a poor guide for the next window (as openai/whisper).
                if options.condition_on_previous_text and result.temperature <= 0.5:
                    previous = (previous + [t for seg in result.segments for t in seg.tokens])[-MAX_PROMPT_TOKENS:]
                else:
                    previous = []
            seek += max(int(result.consumed * SAMPLE_RATE), SAMPLE_RATE)  # always move forward
            if on_progress:
                on_progress(min(1.0, seek / audio.shape[0]))

        return TranscriptionResult(language=detected or language, segments=segments)

    def detect_language(self, encoded: np.ndarray) -> str | None:
        if not self.tokens.multilingual:
            return None
        return self._language_from(self._next_logits([self.tokens.sot], encoded))

    def _language_from(self, logits: np.ndarray) -> str | None:
        if not self.tokens.multilingual:
            return None
        codes = list(self.tokens.lang_to_id)
        ids = np.array([self.tokens.lang_to_id[c] for c in codes])
        return codes[int(np.argmax(logits[ids]))]

    def _no_speech_prob(self, logits: np.ndarray) -> float | None:
        if self.tokens.no_speech is None:
            return None
        log_probs = logits - np.logaddexp.reduce(logits)
        return round(float(np.exp(log_probs[self.tokens.no_speech])), 4)

    # ---------------------------------------------------------------- decoding

    def _encode_prompt(self, text: str | None) -> list[int]:
        if not text or not text.strip() or self.tokens.prev is None:
            return []
        ids = self.tokenizer.encode(" " + text.strip(), add_special_tokens=False).ids
        return [t for t in ids if t < self.tokens.eot][-MAX_PROMPT_TOKENS:]

    def _prompt(self, language: str | None, previous: list[int] | None = None) -> list[int]:
        prompt = [self.tokens.sot]
        if self.tokens.multilingual:
            lang_id = self.tokens.lang_to_id.get(language or "en")
            if lang_id is None:
                raise ValueError(f"Unsupported language '{language}'")
            prompt.append(lang_id)
            if self.tokens.transcribe is not None:
                prompt.append(self.tokens.transcribe)
        if previous and self.tokens.prev is not None:
            prompt = [self.tokens.prev, *previous[-MAX_PROMPT_TOKENS:], *prompt]
        return prompt

    def _next_logits(self, tokens: list[int], encoded: np.ndarray) -> np.ndarray:
        return self._decoder_session(encoded)(tokens)

    def _decoder_session(self, encoded: np.ndarray):
        """Return step(tokens) -> logits for the next token, for one window.

        With the merged decoder, step() feeds only the tokens added since the previous call and keeps the
        self-attention K/V in between; the cross-attention K/V (from encoded) are computed on the first call.
        """
        if not self._kv_cache:
            def full_prefix(tokens: list[int]) -> np.ndarray:
                logits = self.decoder.run(
                    ["logits"],
                    {"input_ids": np.array([tokens], dtype=np.int64), "encoder_hidden_states": encoded},
                )[0]
                return logits[0, -1].astype(np.float32)
            return full_prefix

        past = {name: np.zeros(shape, dtype=np.float32) for name, shape in self._empty_past.items()}
        fed = 0

        def cached(tokens: list[int]) -> np.ndarray:
            nonlocal fed
            first = fed == 0
            outputs = self.decoder.run(None, {
                "input_ids": np.array([tokens[fed:]], dtype=np.int64),
                "encoder_hidden_states": encoded,
                "use_cache_branch": np.array([not first]),
                **past,
            })
            logits = None
            for name, value in zip(self._decoder_outputs, outputs):
                if name == "logits":
                    logits = value
                elif first or ".decoder." in name:  # later calls return placeholder cross-attention K/V
                    past["past_key_values" + name[len("present"):]] = value
            fed = len(tokens)
            return logits[0, -1].astype(np.float32)
        return cached

    def _decode_with_fallback(self, encoded: np.ndarray, language: str | None, window_seconds: float,
                              previous: list[int], no_speech: float | None, options: DecodingOptions,
                              rng: np.random.Generator) -> WindowResult:
        temperatures = FALLBACK_TEMPERATURES if options.temperature_fallback else (0.0,)
        for temperature in temperatures:
            result = self._decode_window(encoded, language, window_seconds, previous, temperature, rng)
            looks_wrong = (result.compression_ratio > COMPRESSION_RATIO_THRESHOLD
                           or result.avg_logprob < LOGPROB_THRESHOLD)
            if not looks_wrong or (no_speech is not None and no_speech > NO_SPEECH_THRESHOLD):
                break  # good enough, or silence (skipped by the caller, retrying won't help)
        return result

    def _decode_window(self, encoded: np.ndarray, language: str | None, window_seconds: float,
                       previous: list[int] | None = None, temperature: float = 0.0,
                       rng: np.random.Generator | None = None) -> WindowResult:
        """Decode one 30 s window: greedy at temperature 0, otherwise sampled."""
        prompt = self._prompt(language, previous)
        tokens = list(prompt)
        logprobs: list[float] = []  # of each chosen token, after the filters (as Whisper scores them)
        next_logits = self._decoder_session(encoded)
        for _ in range(min(self.max_new_tokens, N_TEXT_CTX - len(prompt))):
            logits = next_logits(tokens)
            self._apply_rules(logits, tokens[len(prompt):])
            finite = np.isfinite(logits)
            if temperature > 0:
                scaled = np.where(finite, logits / temperature, -np.inf)
                probs = np.exp(scaled - scaled[finite].max())
                next_token = int((rng or np.random.default_rng()).choice(probs.shape[0], p=probs / probs.sum()))
            else:
                next_token = int(np.argmax(logits))
            logprobs.append(float(logits[next_token] - np.logaddexp.reduce(logits[finite])))
            tokens.append(next_token)
            if next_token == self.tokens.eot:
                break
        sampled = tokens[len(prompt):]
        segments, consumed = self._split_segments(sampled, window_seconds, logprobs)
        text = self._decode_text([t for t in sampled if t < self.tokens.eot]).encode()
        return WindowResult(
            segments=segments,
            consumed=consumed,
            avg_logprob=sum(logprobs) / len(logprobs) if logprobs else 0.0,
            compression_ratio=len(text) / len(zlib.compress(text)) if text else 0.0,
            temperature=temperature,
        )

    def _apply_rules(self, logits: np.ndarray, sampled: list[int]) -> None:
        """Whisper's logit filters: suppress specials and enforce well-formed timestamp pairs."""
        t = self.tokens
        ts = t.timestamp_begin
        logits[t.suppress] = -np.inf
        # Special tokens between <|endoftext|> and the timestamps are never valid output.
        logits[t.eot + 1 : ts] = -np.inf

        if not sampled:
            # First token must be a timestamp near the window start.
            logits[: ts] = -np.inf
            logits[ts + int(MAX_INITIAL_TIMESTAMP / TIME_PRECISION) + 1 :] = -np.inf
            return

        last_was_ts = sampled[-1] >= ts
        penultimate_was_ts = len(sampled) < 2 or sampled[-2] >= ts
        if last_was_ts:
            if penultimate_was_ts:
                logits[ts:] = -np.inf  # a segment needs text after its start timestamp
            else:
                logits[: t.eot] = -np.inf  # after closing a segment: timestamp or end

        timestamps = [tok for tok in sampled if tok >= ts]
        if timestamps:
            # Timestamps never go backwards (a closing tag may be repeated as the next opening).
            floor = timestamps[-1] if last_was_ts and not penultimate_was_ts else timestamps[-1] + 1
            logits[ts:floor] = -np.inf

        # If any timestamp is more likely than all text tokens combined, emit a timestamp.
        log_probs = logits - np.logaddexp.reduce(logits[np.isfinite(logits)])
        ts_mass = np.logaddexp.reduce(log_probs[ts:]) if np.isfinite(log_probs[ts:]).any() else -np.inf
        if ts_mass > np.max(log_probs[:ts]):
            logits[:ts] = -np.inf

    def _split_segments(self, sampled: list[int], window_seconds: float,
                        logprobs: list[float] | None = None) -> tuple[list[Segment], float]:
        t = self.tokens
        if sampled and sampled[-1] == t.eot:
            sampled = sampled[:-1]
            finished = True
        else:
            finished = False  # hit max_new_tokens

        logprobs = logprobs or [0.0] * len(sampled)
        segments: list[Segment] = []
        start: float | None = None
        text_ids: list[int] = []
        text_lps: list[float] = []
        last_end = 0.0

        def mean(values: list[float]) -> float | None:
            return round(sum(values) / len(values), 4) if values else None

        for tok, lp in zip(sampled, logprobs):
            if tok >= t.timestamp_begin:
                time = (tok - t.timestamp_begin) * TIME_PRECISION
                if start is None:
                    start, text_ids, text_lps = time, [], []
                else:
                    segments.append(Segment(start, time, self._decode_text(text_ids), mean(text_lps),
                                            tokens=text_ids))
                    last_end, start = time, None
            elif start is not None:
                text_ids.append(tok)
                text_lps.append(lp)

        if start is not None and text_ids:
            # Unclosed trailing segment: keep it if the model finished, otherwise re-decode from there.
            if finished or not segments:
                segments.append(Segment(start, window_seconds, self._decode_text(text_ids), mean(text_lps),
                                        tokens=text_ids))
                return segments, window_seconds
            return segments, last_end

        if finished or not segments:
            return segments, window_seconds
        return segments, last_end

    def _decode_text(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True).strip()
