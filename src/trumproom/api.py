import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from enum import Enum
from threading import Thread
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from trumproom.crew import Trumproom

DEFAULT_TOPIC = "President Donald Trump"


# --- Models ---


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class RunRequest(BaseModel):
    topic: str = Field(default=DEFAULT_TOPIC, examples=["President Donald Trump"])


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    created_at: str
    topic: str


class JobResult(BaseModel):
    job_id: str
    status: JobStatus
    created_at: str
    completed_at: str | None = None
    topic: str
    result: Any | None = None
    error: str | None = None


# --- In-memory job store ---

_jobs: dict[str, dict] = {}


# --- Crew execution ---


def _run_crew(job_id: str, topic: str) -> None:
    _jobs[job_id]["status"] = JobStatus.RUNNING
    try:
        inputs = {
            "topic": topic,
            "current_date": str(datetime.now(timezone.utc).date()),
        }
        result = Trumproom().crew().kickoff(inputs=inputs)
        _jobs[job_id]["status"] = JobStatus.COMPLETED
        _jobs[job_id]["completed_at"] = datetime.now(timezone.utc).isoformat()
        _jobs[job_id]["result"] = result.raw if hasattr(result, "raw") else str(result)
    except Exception as e:
        _jobs[job_id]["status"] = JobStatus.FAILED
        _jobs[job_id]["completed_at"] = datetime.now(timezone.utc).isoformat()
        _jobs[job_id]["error"] = str(e)


# --- App ---


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    _jobs.clear()


app = FastAPI(
    title="Trumproom API",
    description="Analyze recent statements by public figures and predict stock market effects. NOT financial advice.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/runs", response_model=JobResponse, status_code=202)
async def create_run(request: RunRequest | None = None):
    """Kick off a crew analysis run. Returns immediately with a job ID for polling."""
    topic = request.topic if request else DEFAULT_TOPIC
    job_id = uuid.uuid4().hex[:12]
    now = datetime.now(timezone.utc).isoformat()

    _jobs[job_id] = {
        "job_id": job_id,
        "status": JobStatus.PENDING,
        "created_at": now,
        "topic": topic,
        "result": None,
        "error": None,
        "completed_at": None,
    }

    thread = Thread(target=_run_crew, args=(job_id, topic), daemon=True)
    thread.start()

    return JobResponse(
        job_id=job_id,
        status=JobStatus.PENDING,
        created_at=now,
        topic=topic,
    )


@app.get("/runs/{job_id}", response_model=JobResult)
async def get_run(job_id: str):
    """Poll for the status and result of a crew run."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobResult(**job)


@app.get("/runs", response_model=list[JobResponse])
async def list_runs():
    """List all crew runs."""
    return [
        JobResponse(
            job_id=j["job_id"],
            status=j["status"],
            created_at=j["created_at"],
            topic=j["topic"],
        )
        for j in _jobs.values()
    ]


@app.delete("/runs/{job_id}", status_code=204)
async def delete_run(job_id: str):
    """Remove a completed or failed run from the store."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] in (JobStatus.PENDING, JobStatus.RUNNING):
        raise HTTPException(status_code=409, detail="Cannot delete a running job")
    del _jobs[job_id]


def start():
    """Entry point for `serve` script command."""
    uvicorn.run("trumproom.api:app", host="0.0.0.0", port=8000, reload=True)
