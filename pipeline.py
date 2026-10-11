import os
import subprocess
import json
import time
from openai import OpenAI
from pathlib import Path
from logger import log_info, log_error, log_warning
from db import update_task, add_used_minutes, create_user_clip

TEMP_DIR = Path("temp").resolve()
TEMP_DIR.mkdir(exist_ok=True)

GENAPI_BASE_URL = os.getenv("GENAPI_BASE_URL", "https://proxy.gen-api.ru/v1")
GENAPI_MODEL = "gemini-3-8-flash"

MAX_VIDEO_DURATION = 900

PLANS = {
    "free": {
        "name": "Бесплатно", "price": 0, "minutes": 15,
        "max_clip_duration": 60, "watermark": True, "kinetic_subs": False,
    },
    "pro": {
        "name": "Про", "price": 500, "minutes": 300,
        "max_clip_duration": 180, "watermark": False, "kinetic_subs": True,
    },
    "business": {
        "name": "Бизнес", "price": 1500, "minutes": 1000,
        "max_clip_duration": 300, "watermark": False, "kinetic_subs": True,
    },
    "agency": {
        "name": "Агентство", "price": 5000, "minutes": 5000,
        "max_clip_duration": 900, "watermark": False, "kinetic_subs": True,
    },
}


class PipelineError(Exception):
    pass


def download_video(url: str, output_path: str):
    log_info(f"[download] Начало скачивания: {url}")
    start = time.time()

    if os.path.exists(output_path):
        os.remove(output_path)

    cmd = ["yt-dlp"]

    proxy = os.getenv("YT_PROXY")
    if proxy:
        cmd += ["--proxy", proxy]

    cookies_path = "cookies.txt"
    if os.path.exists(cookies_path):
        cmd += ["--cookies", cookies_path]

    cmd += [
        "-f", "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/bv*+ba/b",
        "--merge-output-format", "mp4",
        "-o", output_path,
        url,
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        duration = time.time() - start
        size_mb = os.path.getsize(output_path) / 1024 / 1024
        log_info(f"[download] Готово за {duration:.1f}с, размер: {size_mb:.1f}MB")
        return output_path
    except subprocess.CalledProcessError as e:
        err = e.stderr.decode('utf-8', errors='ignore')
        if "Video unavailable" in err or "Private video" in err:
            raise PipelineError("Видео недоступно или приватное")
        if "This video is not available" in err:
            raise PipelineError("Видео недоступно в твоём регионе")
        if "Sign in" in err or "bot" in err.lower():
            raise PipelineError("YouTube требует авторизацию. Попробуй позже или с VPN.")
        log_error(f"[download] Ошибка: {err[-500:]}")
        raise PipelineError("Не удалось скачать видео. Проверь ссылку.")


def get_video_duration(video_path: str) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return float(result.stdout.strip())
    except:
        return 0.0


def check_duration_before_download(url: str) -> float:
    cmd = ["yt-dlp", "--no-download", "--print", "%(duration)s", url]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        val = result.stdout.strip().split("\n")[-1]
        return float(val)
    except Exception as e:
        log_warning(f"[check] Не удалось получить длину: {e}")
        return 0.0


def transcribe(video_path: str):
    log_info(f"[transcribe] Начало")
    start = time.time()

    from whisper import load_model
    model = load_model("base")
    result = model.transcribe(video_path, word_timestamps=True)

    log_info(f"[transcribe] Готово за {time.time()-start:.1f}с, сегментов: {len(result['segments'])}")
    return result


def calculate_num_clips(video_duration: float, max_clip_duration: int = 180) -> int:
    if video_duration < 30:
        return 1
    elif video_duration < 60:
        return 2
    elif video_duration < 180:
        return 5
    elif video_duration < 600:
        return 8
    else:
        return 10


def find_interesting_moments(transcript: str, num_clips: int,
                              video_duration: float, max_clip_duration: int):
    log_info(f"[gemini] Анализ, клипов: {num_clips}, макс длина: {max_clip_duration}с")
    start = time.time()

    api_key = os.getenv("GENAPI_API_KEY")
    if not api_key:
        raise PipelineError("API-ключ не настроен на сервере")

    client = OpenAI(api_key=api_key, base_url=GENAPI_BASE_URL)

    duration_int = int(video_duration)
    max_dur = min(max_clip_duration, duration_int)

    if num_clips == 1:
        prompt = f"""Ты — редактор вирусных видео для TikTok и Shorts.
Длина видео: {duration_int} сек. Макс длина клипа: {max_dur} сек.
Верни ОДИН момент — самое интересное место.
JSON без markdown: [{{"start": 0.0, "end": {min(max_dur, duration_int)}, "title": "Заголовок"}}]
Транскрипт:
{transcript}"""
    else:
        prompt = f"""Ты — редактор вирусных видео для TikTok и Shorts.
Длина видео: {duration_int} сек. Макс длина клипа: {max_dur} сек.
ВАЖНО: start и end не превышают {duration_int}. Длина клипа не превышает {max_dur} сек.
Найди {num_clips} самых интересных моментов.
JSON без markdown:
[{{"start": 12.5, "end": 45.3, "title": "Заголовок"}}]
Транскрипт:
{transcript}"""

    try:
        response = client.chat.completions.create(
            model=GENAPI_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.choices[0].message.content.strip()
        text = text.replace("```json", "").replace("```", "").strip()
        moments = json.loads(text)

        cleaned = []
        for m in moments:
            start_t = max(0.0, min(m["start"], video_duration))
            end_t = max(0.0, min(m["end"], video_duration))
            if end_t - start_t > max_clip_duration:
                end_t = start_t + max_clip_duration
            if end_t - start_t >= 5:
                cleaned.append({
                    "start": start_t,
                    "end": end_t,
                    "title": m.get("title", "Клип"),
                })

        log_info(f"[gemini] Готово за {time.time()-start:.1f}с, моментов: {len(cleaned)}")
        return cleaned
    except json.JSONDecodeError:
        raise PipelineError("AI вернул некорректный ответ. Попробуй ещё раз.")
    except Exception as e:
        log_error(f"[gemini] Ошибка: {e}")
        raise PipelineError("Ошибка AI-анализа.")


def format_ass_time(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int((seconds - int(seconds)) * 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def generate_ass(segments, output_path, clip_start, clip_end,
                 use_watermark=False, kinetic_subs=True):
    header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial Black,72,&H00FFFFFF,&H0000FFFF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,6,0,2,80,80,300,1
Style: Watermark,Arial,56,&H50FFFFFF,&H50FFFFFF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,2,0,9,40,40,40,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events = []

    if use_watermark:
        total_duration = clip_end - clip_start
        events.append(
            f"Dialogue: 0,0:00:00.00,{format_ass_time(total_duration)},Watermark,,0,0,0,,SonarClip"
        )

    for seg in segments:
        if seg["end"] <= clip_start or seg["start"] >= clip_end:
            continue
        if "words" not in seg or not seg["words"]:
            continue

        words = [
            w for w in seg["words"]
            if w["end"] > clip_start and w["start"] < clip_end
        ]
        if not words:
            continue

        if kinetic_subs:
            for i in range(0, len(words), 2):
                group = words[i:i + 2]
                if not group:
                    continue
                for j, active in enumerate(group):
                    w_start = max(active["start"], clip_start) - clip_start
                    w_end = min(active["end"], clip_end) - clip_start

                    parts = []
                    for k, w in enumerate(group):
                        t = w["word"].strip().upper()
                        if k == j:
                            parts.append(f"{{\\c&H00FFFF&\\fscx115\\fscy115}}{t}{{\\r}}")
                        else:
                            parts.append(t)

                    events.append(
                        f"Dialogue: 0,{format_ass_time(w_start)},{format_ass_time(w_end)},"
                        f"Default,,0,0,0,,{' '.join(parts)}"
                    )
        else:
            seg_start = max(seg["start"], clip_start) - clip_start
            seg_end = min(seg["end"], clip_end) - clip_start
            text = seg["text"].strip().upper()
            if text:
                events.append(
                    f"Dialogue: 0,{format_ass_time(seg_start)},{format_ass_time(seg_end)},"
                    f"Default,,0,0,0,,{text}"
                )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(header)
        f.write("\n".join(events))
    return output_path


def cut_clip(video_path, start, end, output_path, ass_path=None):
    duration = end - start
    filters = [
        "scale=1080:1920:force_original_aspect_ratio=decrease",
        "pad=1080:1920:(ow-iw)/2:(oh-ih)/2:color=black",
    ]
    if ass_path and os.path.exists(ass_path):
        ass_abs = str(Path(ass_path).resolve())
        ass_escaped = ass_abs.replace("\\", "/").replace(":", "\\:")
        filters.append(f"ass='{ass_escaped}'")

    cmd = [
        "ffmpeg", "-i", video_path,
        "-ss", str(start), "-t", str(duration),
        "-vf", ",".join(filters),
        "-c:v", "libx264", "-preset", "fast",
        "-c:a", "aac", "-y", output_path,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return output_path
    except subprocess.CalledProcessError as e:
        err = e.stderr.decode('utf-8', errors='ignore')
        log_error(f"[ffmpeg] Ошибка: {err[-500:]}")
        raise PipelineError("Ошибка нарезки видео")


def generate_thumb(video_path: str, output_path: str, at: float = 0.5):
    cmd = [
        "ffmpeg", "-i", video_path,
        "-ss", str(at),
        "-vframes", "1",
        "-vf", "scale=360:640",
        "-y", output_path,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return output_path
    except subprocess.CalledProcessError:
        return None


def run_pipeline_with_task(task_id: str, url: str, plan: str, user_id: str):
    plan_config = PLANS.get(plan, PLANS["free"])
    max_clip_duration = plan_config["max_clip_duration"]
    use_watermark = plan_config["watermark"]
    kinetic_subs = plan_config["kinetic_subs"]

    try:
        update_task(task_id, status="processing", progress=5, stage="Проверяю видео...")

        pre_duration = check_duration_before_download(url)
        if pre_duration > MAX_VIDEO_DURATION:
            raise PipelineError(
                f"Видео длиннее {MAX_VIDEO_DURATION // 60} минут. Выбери другое."
            )

        update_task(task_id, progress=15, stage="Скачиваю видео...")
        video_path = str(TEMP_DIR / "source.mp4")
        download_video(url, video_path)

        video_duration = get_video_duration(video_path)
        log_info(f"[task {task_id}] Длина видео: {video_duration:.1f}с")

        num_clips = calculate_num_clips(video_duration, max_clip_duration)

        update_task(task_id, progress=30, stage="Транскрибирую...")
        result = transcribe(video_path)
        transcript = result["text"]
        segments = result["segments"]

        update_task(task_id, progress=60, stage="Анализирую через Gemini...")
        moments = find_interesting_moments(
            transcript, num_clips, video_duration, max_clip_duration
        )

        update_task(task_id, progress=75, stage="Нарезаю клипы...")
        clips = []
        total_minutes = 0.0

        for i, moment in enumerate(moments):
            output = str(TEMP_DIR / f"clip_{i+1}.mp4")
            ass_path = str(TEMP_DIR / f"clip_{i+1}.ass")
            thumb_path = str(TEMP_DIR / f"clip_{i+1}.jpg")

            generate_ass(segments, ass_path, moment["start"], moment["end"],
                         use_watermark=use_watermark, kinetic_subs=kinetic_subs)
            cut_clip(video_path, moment["start"], moment["end"], output, ass_path)
            generate_thumb(output, thumb_path, at=0.5)

            clip_minutes = (moment["end"] - moment["start"]) / 60
            total_minutes += clip_minutes

            progress = 75 + int((i + 1) / len(moments) * 20)
            update_task(task_id, progress=progress, stage=f"Клип {i+1}/{len(moments)}")

            clip_db = create_user_clip(
                user_id=user_id,
                task_id=task_id,
                title=moment["title"],
                start_time=moment["start"],
                end_time=moment["end"],
                filename=f"clip_{i+1}.mp4",
                thumb_filename=f"clip_{i+1}.jpg" if os.path.exists(thumb_path) else None,
            )
            if clip_db is None:
                log_error(f"[task {task_id}] Не удалось сохранить клип {i+1} в БД")

            clips.append({
                "id": i + 1,
                "title": moment["title"],
                "start": moment["start"],
                "end": moment["end"],
                "path": output,
                "thumb": thumb_path,
            })

        add_used_minutes(user_id, total_minutes)

        update_task(
            task_id,
            status="completed",
            progress=100,
            stage="Готово",
            clips=clips,
        )

        log_info(f"[task {task_id}] Завершено. Клипов: {len(clips)}, минут: {total_minutes:.1f}")

    except PipelineError as e:
        log_error(f"[task {task_id}] PipelineError: {e}")
        update_task(task_id, status="error", error=str(e), stage="Ошибка")
    except Exception as e:
        log_error(f"[task {task_id}] Ошибка: {e}")
        update_task(task_id, status="error", error="Внутренняя ошибка сервера", stage="Ошибка")