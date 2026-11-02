from fastapi import FastAPI, HTTPException, Header, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path
from typing import Optional, List
import os
import time
import threading
import uuid
import shutil
from pipeline import run_pipeline_with_task, TEMP_DIR, UPLOAD_DIR, PLANS
from db import (
    get_profile, get_profile_by_email, create_profile,
    create_task, get_task, get_active_task,
    get_user_clips, delete_user_clips,
)
from auth import hash_password, verify_password, create_token, decode_token
from logger import log_info, log_error

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)


class RegisterRequest(BaseModel):
    email: str
    password: str
    name: str = ""


class LoginRequest(BaseModel):
    email: str
    password: str


class ProcessUrlRequest(BaseModel):
    url: str


class DeleteClipsRequest(BaseModel):
    clip_ids: List[str]


def get_current_user(authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Не авторизован")
    token = authorization.replace("Bearer ", "")
    payload = decode_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Токен недействителен")
    return payload


def profile_to_dict(profile: dict):
    return {
        "id": str(profile["id"]),
        "email": profile["email"],
        "name": profile.get("name"),
        "plan": profile.get("plan"),
        "used_minutes": profile.get("used_minutes"),
        "minutes_limit": profile.get("minutes_limit"),
    }


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/register")
def register(req: RegisterRequest):
    log_info(f"[auth] Регистрация: {req.email}")

    if len(req.password) < 6:
        raise HTTPException(status_code=400, detail="Пароль минимум 6 символов")

    if get_profile_by_email(req.email):
        raise HTTPException(status_code=400, detail="Email уже занят")

    password_hash = hash_password(req.password)
    profile = create_profile(req.email, password_hash, req.name)
    if not profile:
        raise HTTPException(status_code=500, detail="Не удалось создать профиль")

    token = create_token(str(profile["id"]), profile["email"])
    return {"token": token, "user": profile_to_dict(profile)}


@app.post("/login")
def login(req: LoginRequest):
    log_info(f"[auth] Логин: {req.email}")

    profile = get_profile_by_email(req.email)
    if not profile:
        raise HTTPException(status_code=401, detail="Неверный email или пароль")

    if not verify_password(req.password, profile["password_hash"]):
        raise HTTPException(status_code=401, detail="Неверный email или пароль")

    token = create_token(str(profile["id"]), profile["email"])
    return {"token": token, "user": profile_to_dict(profile)}


@app.get("/me")
def me(authorization: Optional[str] = Header(None)):
    payload = get_current_user(authorization)
    profile = get_profile(payload["sub"])
    if not profile:
        raise HTTPException(status_code=404, detail="Профиль не найден")
    return profile_to_dict(profile)


@app.get("/tasks/active")
def active_task(authorization: Optional[str] = Header(None)):
    payload = get_current_user(authorization)
    task = get_active_task(payload["sub"])
    if not task:
        return {"task": None}
    return {
        "task": {
            "id": str(task["id"]),
            "status": task["status"],
            "progress": task["progress"],
            "stage": task["stage"],
            "error": task.get("error"),
            "clips": task.get("clips") or [],
        }
    }


@app.post("/upload")
async def upload_video(
    file: UploadFile = File(...),
    mirror: bool = Form(False),
    speed_up: bool = Form(False),
    subtitles: bool = Form(True),
    authorization: Optional[str] = Header(None),
):
    payload = get_current_user(authorization)
    user_id = payload["sub"]

    log_info(f"[api] POST /upload | user={user_id} | file={file.filename}")

    profile = get_profile(user_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Профиль не найден")

    user_plan = profile.get("plan", "free")
    plan_config = PLANS.get(user_plan, PLANS["free"])
    used = profile.get("used_minutes") or 0
    limit = plan_config["minutes"]

    if used >= limit:
        raise HTTPException(status_code=403, detail=f"Лимит исчерпан. Использовано {used:.1f} из {limit} минут.")

    file_id = str(uuid.uuid4())
    ext = Path(file.filename).suffix or ".mp4"
    save_path = str(UPLOAD_DIR / f"{file_id}{ext}")

    try:
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        log_error(f"[upload] Save error: {e}")
        raise HTTPException(status_code=500, detail="Не удалось сохранить файл")

    file_size_mb = os.path.getsize(save_path) / 1024 / 1024
    log_info(f"[upload] Сохранён: {save_path} ({file_size_mb:.1f}MB)")

    from pipeline import get_video_duration
    video_duration = get_video_duration(save_path)
    duration_minutes = video_duration / 60

    log_info(f"[upload] Длина: {duration_minutes:.1f} мин")

    if duration_minutes > plan_config["max_file_minutes"]:
        os.remove(save_path)
        raise HTTPException(
            status_code=400,
            detail=f"Файл длиннее {plan_config['max_file_minutes']} минут. Тариф: {plan_config['name']}."
        )

    if used + duration_minutes > limit:
        os.remove(save_path)
        remaining = limit - used
        raise HTTPException(
            status_code=403,
            detail=f"Не хватит минут. Осталось: {remaining:.1f}, нужно: {duration_minutes:.1f}."
        )

    task = create_task(user_id, f"file:{file.filename}", user_plan)
    if not task:
        os.remove(save_path)
        raise HTTPException(status_code=500, detail="Не удалось создать задачу")

    task_id = str(task["id"])

    def run_async():
        try:
            run_pipeline_with_task(task_id, save_path, user_plan, user_id,
                                    mirror=mirror, speed_up=speed_up, subtitles=subtitles)
        except Exception as e:
            log_error(f"[async] {e}")

    threading.Thread(target=run_async, daemon=True).start()

    return {
        "task": {
            "id": task_id,
            "status": task["status"],
            "progress": task["progress"],
            "stage": task["stage"],
        }
    }


@app.get("/status/{task_id}")
def status(task_id: str, authorization: Optional[str] = Header(None)):
    payload = get_current_user(authorization)
    task = get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    if str(task["user_id"]) != payload["sub"]:
        raise HTTPException(status_code=403, detail="Нет доступа")
    return {
        "id": str(task["id"]),
        "status": task["status"],
        "progress": task["progress"],
        "stage": task["stage"],
        "error": task.get("error"),
        "clips": task.get("clips") or [],
    }


@app.get("/clips")
def list_user_clips(authorization: Optional[str] = Header(None)):
    payload = get_current_user(authorization)
    clips = get_user_clips(payload["sub"])
    return {
        "clips": [
            {
                "id": str(c["id"]),
                "title": c["title"],
                "start": c["start_time"],
                "end": c["end_time"],
                "filename": c["filename"],
                "thumb": c["thumb_filename"],
                "created_at": c["created_at"].isoformat() if c.get("created_at") else None,
            }
            for c in clips
        ]
    }


@app.delete("/clips")
def delete_clips(req: DeleteClipsRequest, authorization: Optional[str] = Header(None)):
    payload = get_current_user(authorization)
    rows = delete_user_clips(payload["sub"], req.clip_ids)

    for r in rows:
        f = TEMP_DIR / r["filename"]
        if f.exists():
            f.unlink()
        if r.get("thumb_filename"):
            t = TEMP_DIR / r["thumb_filename"]
            if t.exists():
                t.unlink()

    return {"deleted": len(rows)}


@app.get("/clip/{filename}")
def get_clip(filename: str):
    path = TEMP_DIR / Path(filename).name
    if not path.exists():
        raise HTTPException(status_code=404, detail="Clip not found")
    return FileResponse(path, media_type="video/mp4")


@app.get("/thumb/{filename}")
def get_thumb(filename: str):
    path = TEMP_DIR / Path(filename).name
    if not path.exists():
        raise HTTPException(status_code=404, detail="Thumb not found")
    return FileResponse(path, media_type="image/jpeg")