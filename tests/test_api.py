import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.audio import InvalidAudioError
from app.main import Components, create_app
from tests.conftest import FakeTranscriber, make_wav, wait_for


def upload(client, path="/jobs", data=None, content=None, headers=None, params=None):
    files = {"file": ("episode.wav", content if content is not None else make_wav(), "audio/wav")}
    return client.post(path, files=files, data=data or {}, headers=headers, params=params)


def test_transcribe_returns_json_transcript(client, transcriber):
    response = upload(client, "/transcribe", data={"language": "en"})
    assert response.status_code == 200
    body = response.json()
    assert body["language"] == "en"
    assert body["model"] == "fake-whisper"
    assert body["duration_seconds"] == 1.0
    assert body["text"].startswith("Welcome to the show")
    assert [s["index"] for s in body["segments"]] == [0, 1, 2]
    assert body["segments"][1] == {"index": 1, "start": 4.2, "end": 9.8, "avg_logprob": -0.3, "no_speech_prob": 0.01,
                                   "text": "Today we talk about ONNX Runtime and model deployment."}
    assert body["metrics"]["audio_seconds"] == 1.0
    assert transcriber.languages == ["en"]
    assert client.get("/jobs").json() == []  # synchronous requests leave no job behind


def test_transcribe_auto_language_by_default(client, transcriber):
    assert upload(client, "/transcribe").status_code == 200
    assert transcriber.languages == [None]  # None = detect


@pytest.mark.parametrize("fmt,media_type,expected", [
    ("text", "text/plain", "Welcome to the show about machine learning. Today we talk"),
    ("srt", "application/x-subrip", "1\n00:00:00,000 --> 00:00:04,200\nWelcome to the show about machine learning.\n"),
    ("vtt", "text/vtt", "WEBVTT\n\n00:00:00.000 --> 00:00:04.200\nWelcome"),
])
def test_transcribe_output_formats(client, fmt, media_type, expected):
    response = upload(client, "/transcribe", params={"format": fmt})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith(media_type)
    assert response.text.startswith(expected)
    assert "episode." in response.headers["content-disposition"]


def test_job_lifecycle(client):
    response = upload(client, data={"language": "de"})
    assert response.status_code == 202
    job = response.json()
    assert job["status_url"] == f"/jobs/{job['id']}"

    status = wait_for(client, job["id"])
    assert status["status"] == "completed"
    assert status["model"] == "fake-whisper"
    assert status["metrics"]["real_time_factor"] >= 0
    assert "transcript" not in status

    transcript = client.get(job["transcript_url"]).json()
    assert transcript["language"] == "de" and len(transcript["segments"]) == 3
    assert client.get(job["transcript_url"], params={"format": "text"}).text.endswith("next week.\n")
    assert [j["id"] for j in client.get("/jobs").json()] == [job["id"]]

    assert client.delete(f"/jobs/{job['id']}").status_code == 204
    assert client.get(f"/jobs/{job['id']}").status_code == 404


def test_transcript_not_ready_is_409(client, transcriber):
    transcriber.gate = threading.Event()
    job = upload(client).json()
    try:
        response = client.get(job["transcript_url"])
        assert response.status_code == 409 and "not ready" in response.json()["detail"]
    finally:
        transcriber.gate.set()
    assert wait_for(client, job["id"])["status"] == "completed"


def test_upload_is_deleted_after_transcription(client, settings):
    job = upload(client).json()
    wait_for(client, job["id"])
    assert list(Path(settings.upload_dir).iterdir()) == []


def test_failed_transcription(settings):
    transcriber = FakeTranscriber(error=InvalidAudioError("broken stream"))
    with TestClient(create_app(settings, Components(transcriber=transcriber))) as client:
        job = upload(client).json()
        status = wait_for(client, job["id"])
        assert status["status"] == "failed" and status["error"] == "broken stream"
        assert client.get(job["transcript_url"]).status_code == 422

        response = upload(client, "/transcribe")
        assert response.status_code == 422 and "broken stream" in response.json()["detail"]


def test_rejects_non_audio(client):
    response = upload(client, content=b"%PDF-1.7 not audio")
    assert response.status_code == 415


def test_rejects_empty_and_oversized(client):
    assert upload(client, content=b"").status_code == 400
    big = b"RIFF\x00\x00\x00\x00WAVE" + b"\x00" * (2 * 1024 * 1024)
    assert upload(client, content=big).status_code == 413


def test_rejects_too_long_audio(client):
    status = wait_for(client, upload(client, content=make_wav(seconds=65, rate=4000)).json()["id"])
    assert status["status"] == "failed" and "60s limit" in status["error"]


def test_rejects_unsupported_language(client):
    response = upload(client, data={"language": "fr"})
    assert response.status_code == 422 and "/languages" in response.json()["detail"]
    assert upload(client, data={"language": "english"}).status_code == 422  # not an ISO code


def test_languages_and_health(client):
    assert client.get("/languages").json() == {"languages": ["en", "de", "it"], "default": "auto"}
    health = client.get("/health").json()
    assert health["status"] == "ok" and health["model"] == "fake-whisper"
    assert client.get("/healthz").status_code == 200


def test_degraded_without_model(settings, tmp_path):
    settings.whisper_model_dir = str(tmp_path / "missing")
    with TestClient(create_app(settings, Components())) as client:
        health = client.get("/health").json()
        assert health["status"] == "degraded" and "export_whisper_onnx" in health["model_error"]
        assert client.get("/healthz").status_code == 503
        assert upload(client, "/transcribe").status_code == 503


def test_queue_full(settings):
    settings.max_queue_size = 1
    transcriber = FakeTranscriber()
    transcriber.gate = threading.Event()
    with TestClient(create_app(settings, Components(transcriber=transcriber))) as client:
        try:
            first = upload(client).json()  # taken by the worker, held by the gate
            for _ in range(100):
                if transcriber.calls:
                    break
                time.sleep(0.02)
            assert upload(client).status_code == 202  # fills the queue
            response = upload(client)
            assert response.status_code == 503 and response.headers["retry-after"] == "30"
        finally:
            transcriber.gate.set()
        assert wait_for(client, first["id"])["status"] == "completed"


def test_finished_jobs_are_pruned(settings):
    settings.max_finished_jobs = 2
    with TestClient(create_app(settings, Components(transcriber=FakeTranscriber()))) as client:
        ids = [upload(client).json()["id"] for _ in range(3)]
        wait_for(client, ids[-1])
        deadline = time.monotonic() + 5
        while len(client.get("/jobs").json()) > 2 and time.monotonic() < deadline:
            time.sleep(0.02)  # pruning runs right after the job reports completed
        assert client.get(f"/jobs/{ids[0]}").status_code == 404
        assert [j["id"] for j in client.get("/jobs").json()] == ids[:0:-1]


def test_api_token(settings, transcriber):
    settings.api_token = "secret"
    with TestClient(create_app(settings, Components(transcriber=transcriber))) as client:
        assert upload(client, "/transcribe").status_code == 401
        assert client.get("/jobs").status_code == 401
        assert client.get("/languages").status_code == 401
        assert upload(client, "/transcribe", headers={"X-API-Key": "wrong"}).status_code == 401
        assert upload(client, "/transcribe", headers={"X-API-Key": "secret"}).status_code == 200
        assert upload(client, "/transcribe", headers={"Authorization": "bearer secret"}).status_code == 200
        assert client.get("/health").status_code == 200
        assert client.get("/healthz").status_code == 200


def test_progress_and_queue_position(client, transcriber):
    transcriber.gate = threading.Event()
    first, second = upload(client).json(), upload(client).json()
    try:
        for _ in range(100):
            running = client.get(first["status_url"]).json()
            if running["stage"] == "transcribing":
                break
            time.sleep(0.02)
        assert running["status"] == "processing" and running["progress"] == 0.5
        waiting = client.get(second["status_url"]).json()
        assert waiting["status"] == "queued" and waiting["queue_position"] == 0 and waiting["progress"] == 0
    finally:
        transcriber.gate.set()
    done = wait_for(client, first["id"])
    assert done["progress"] == 1.0 and done["stage"] is None and done["queue_position"] is None
    assert wait_for(client, second["id"])["status"] == "completed"


def test_config_reports_the_pipeline(client):
    body = client.get("/config").json()
    assert body["audio"]["sample_rate"] == 16000 and body["windows"]["samples"] == 480000
    assert body["features"]["frame_ms"] == 25 and body["features"]["hop_ms"] == 10
    assert body["features"]["shape"] == [80, 3000] and body["model"]["name"] == "fake-whisper"
    assert body["decoding"]["temperatures"][0] == 0.0


def test_config_needs_the_token(settings, transcriber):
    settings.api_token = "secret"
    with TestClient(create_app(settings, Components(transcriber=transcriber))) as c:
        assert c.get("/config").status_code == 401
        assert c.get("/config", headers={"X-API-Key": "secret"}).status_code == 200


def test_precision_must_be_in_the_export(client):
    # The fake model reports no variants: only the default can be used.
    res = upload(client, data={"encoder": "int8"})
    assert res.status_code == 422 and "encoder precision int8 isn't available" in res.json()["detail"]
    assert upload(client, data={"decoder": "fp16"}).status_code == 422  # not a precision at all
