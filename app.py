"""MaskFlow local MVP: upload a video, describe a masking rule, get the result."""
import hashlib
import queue
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

import ling
import pipeline

ROOT = Path(__file__).parent
DATA = ROOT / "data"
MAX_UPLOAD_BYTES = 200 * 1024 * 1024
ALLOWED_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

app = FastAPI(title="MaskFlow")
jobs: dict = {}
work: queue.Queue = queue.Queue()


def _digest(path: Path) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as stream:
        while chunk := stream.read(1 << 20):
            sha.update(chunk)
    return sha.hexdigest()


def _run(job: dict) -> None:
    def stage(name, progress=None):
        job["stage"] = name
        job["progress"] = progress

    video, output = Path(job["input"]), Path(job["output"])
    stage("Ling-3.2-flash 生成规则代码")
    started = time.perf_counter()
    job["code"] = ling.generate_rule(job["rule"])
    job["stats"]["ling_seconds"] = round(time.perf_counter() - started, 1)

    stage("GPU 检测与跟踪", 0)
    started = time.perf_counter()
    detections, cache_hit = pipeline.load_or_detect(
        video, _digest(video), lambda p: stage("GPU 检测与跟踪", p))
    job["stats"]["detect_seconds"] = round(time.perf_counter() - started, 1)
    job["stats"]["cache_hit"] = cache_hit

    stage("沙箱执行规则")
    decisions = pipeline.apply_rule(job["code"], detections["items"])

    stage("打码并用 NVENC 编码", 0)
    started = time.perf_counter()
    masked = pipeline.render(video, output, detections, decisions,
                             lambda p: stage("打码并用 NVENC 编码", p))
    job["stats"].update(
        render_seconds=round(time.perf_counter() - started, 1),
        frames=detections["frames"],
        detections=len(detections["items"]),
        masked=masked,
    )


def _worker() -> None:
    # One worker on purpose: the GPU only has room for one job at a time.
    while True:
        job = jobs[work.get()]
        job["status"] = "running"
        try:
            _run(job)
            job["status"] = "done"
        except Exception as exc:  # shown to the local user on the job page
            job["status"] = "failed"
            job["error"] = str(exc)
        job["stage"] = None
        job["progress"] = None


@app.on_event("startup")
def _start_worker() -> None:
    threading.Thread(target=_worker, daemon=True).start()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/system")
def system() -> dict:
    return {"device": pipeline.device_name()}


@app.post("/api/jobs")
async def create_job(rule: str = Form(...), video: UploadFile = File(...)) -> dict:
    rule = rule.strip()
    suffix = Path(video.filename or "").suffix.lower()
    if not rule or len(rule) > 500:
        raise HTTPException(400, "规则不能为空，且不超过 500 字")
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(400, "仅支持 mp4、mov、avi、mkv、webm")

    job_id = uuid.uuid4().hex
    folder = DATA / "jobs" / job_id
    folder.mkdir(parents=True)
    source, size = folder / f"input{suffix}", 0
    with open(source, "wb") as stream:
        while chunk := await video.read(1 << 20):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                stream.close()
                source.unlink()
                raise HTTPException(413, "视频不能超过 200MB")
            stream.write(chunk)

    jobs[job_id] = {
        "id": job_id, "rule": rule, "status": "queued", "stage": "排队中",
        "progress": None, "code": None, "error": None, "stats": {},
        "input": str(source), "output": str(folder / "output.mp4"),
    }
    work.put(job_id)
    return {"id": job_id}


def _job(job_id: str) -> dict:
    if job_id not in jobs:
        raise HTTPException(404, "任务不存在")
    return jobs[job_id]


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    return {k: v for k, v in _job(job_id).items() if k not in ("input", "output")}


@app.get("/api/jobs/{job_id}/input")
def get_input(job_id: str) -> FileResponse:
    return FileResponse(_job(job_id)["input"])


@app.get("/api/jobs/{job_id}/output")
def get_output(job_id: str) -> FileResponse:
    job = _job(job_id)
    if job["status"] != "done":
        raise HTTPException(409, "任务尚未完成")
    return FileResponse(job["output"], media_type="video/mp4", filename="maskflow.mp4")


if __name__ == "__main__":
    import uvicorn

    # The sandbox has no OS-level isolation, so stay on localhost.
    uvicorn.run(app, host="127.0.0.1", port=8000)
