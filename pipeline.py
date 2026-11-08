import os
import subprocess
import json
import time
import base64
import shutil
from openai import OpenAI
from pathlib import Path
from logger import log_info, log_error, log_warning
from db import update_task, add_used_minutes, create_user_clip

# --- НАСТРОЙКА FFMPEG ДЛЯ RAILWAY ---
FFMPEG_EXE = "ffmpeg"
try:
    import imageio_ffmpeg
    FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()
    os.environ["IMAGEIO_FFMPEG_EXE"] = FFMPEG_EXE
    log_info(f"[init] FFmpeg найден через imageio: {FFMPEG_EXE}")
    
    # Создаем симлинк, чтобы Whisper (который вызывает 'ffmpeg' напрямую) его увидел
    if not os.path.exists("/usr/local/bin/ffmpeg"):
        try:
            os.symlink(FFMPEG_EXE, "/usr/local/bin/ffmpeg")
            log_info("[init] Создан симлинк /usr/local/bin/ffmpeg")
        except Exception as e:
            log_warning(f"[init] Не удалось создать симлинк: {e}")
            
except Exception as e:
    log_warning(f"[init] imageio-ffmpeg не сработал: {e}")
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        FFMPEG_EXE = system_ffmpeg
        log_info(f"[init] Использую системный FFmpeg: {FFMPEG_EXE}")
    else:
        log_error("[init] FFmpeg НЕ НАЙДЕН!")

TEMP_DIR = Path("/tmp/sonarclip_temp")
TEMP_DIR.mkdir(parents=True, exist_ok=True)

UPLOAD_DIR = Path("/data/uploads") if Path("/data").exists() else Path("/tmp/sonarclip_uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

GENAPI_BASE_URL = os.getenv("GENAPI_BASE_URL", "https://proxy.gen-api.ru/v1")
GENAPI_MODEL = "gemini-3-8-flash"

PLANS = {
    "free": {
        "name": "Бесплатно", "price": 0, "minutes": 15,
        "max_clip_duration": 60, "watermark": True, "kinetic_subs": False,
        "max_file_minutes": 15, "max_clips": 2, "allow_url": False,
    },
    "start": {
        "name": "Старт", "price": 100, "minutes": 40,
        "max_clip_duration": 60, "watermark": False, "kinetic_subs": True,
        "max_file_minutes": 40, "max_clips": 10, "allow_url": False,
    },
    "pro": {
        "name": "Про", "price": 500, "minutes": 300,
        "max_clip_duration": 180, "watermark": False, "kinetic_subs": True,
        "max_file_minutes": 60, "max_clips": 15, "allow_url": True,
    },
    "business": {
        "name": "Бизнес", "price": 1500, "minutes": 1000,
        "max_clip_duration": 300, "watermark": False, "kinetic_subs": True,
        "max_file_minutes": 120, "max_clips": 20, "allow_url": True,
    },
    "agency": {
        "name": "Агентство", "price": 5000, "minutes": 5000,
        "max_clip_duration": 900, "watermark": False, "kinetic_subs": True,
        "max_file_minutes": 300, "max_clips": 40, "allow_url": True,
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

    cookies_b64 = os.getenv("YT_COOKIES_BASE64")
    if cookies_b64:
        try:
            cookies_data = base64.b64decode(cookies_b64).decode('utf-8')
            cookies_path = str(TEMP_DIR / "cookies.txt")
            with open(cookies_path, "w", encoding="utf-8") as f:
                f.write(cookies_data)
            cmd += ["--cookies", cookies_path]
            log_info("[download] Cookies decoded from Base64")
        except Exception as e:
            log_error(f"[download] Cookies decode error: {e}")

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
        if "Sign in" in err or "bot" in err.lower():
            raise PipelineError("YouTube требует авторизацию.")
        log_error(f"[download] Ошибка: {err[-500:]}")
        raise PipelineError("Не удалось скачать видео.")


def get_video_duration(video_path: str) -> float:
    try:
        cmd = [FFMPEG_EXE, "-i", video_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        for line in result.stderr.split('\n'):
            if 'Duration:' in line:
                time_str = line.split('Duration:')[1].split(',')[0].strip()
                h, m, s = time_str.split(':')
                return int(h) * 3600 + int(m) * 60 + float(s)
        return 0.0
    except Exception as e:
        log_error(f"[duration] Ошибка: {e}")
        return 0.0


def transcribe(video_path: str):
    log_info(f"[transcribe] Начало")
    start = time.time()

    from whisper import load_model
    model = load_model("base")
    
    result = model.transcribe(
        video_path, 
        word_timestamps=True,
        condition_on_previous_text=False,
        language="ru" 
    )

    log_info(f"[transcribe] Готово за {time.time()-start:.1f}с, сегментов: {len(result['segments'])}")
    return result


def calculate_num_clips(video_duration: float, max_clips: int) -> int:
    if video_duration < 60:
        return min(2, max_clips)
    elif video_duration < 300:
        return min(5, max_clips)
    elif video_duration < 600:
        return min(10, max_clips)
    elif video_duration < 1800:
        return min(15, max_clips)
    else:
        return min(25, max_clips)


def find_interesting_moments(transcript: str, num_clips: int,
                              video_duration: float, max_clip_duration: int):
    log_info(f"[gemini] Анализ, клипов: {num_clips}, макс длина: {max_clip_duration}с")
    start = time.time()

    api_key = os.getenv("GENAPI_API_KEY")
    if not api_key:
        raise PipelineError("API-ключ не настроен")

    client = OpenAI(api_key=api_key, base_url=GENAPI_BASE_URL)

    duration_int = int(video_duration)
    max_dur = min(max_clip_duration, duration_int)

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


def generate_ass(segments, output_path, clip_start, clip_end, use_watermark=False):
    header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial Black,64,&H00FFFFFF,&H0000FFFF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,6,0,2,80,80,400,1
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

    all_words = []
    for seg in segments:
        if "words" not in seg or not seg["words"]:
            continue
        
        for w in seg["words"]:
            if w["end"] > clip_start and w["start"] < clip_end:
                w_start = max(w["start"], clip_start) - clip_start
                w_end = min(w["end"], clip_end) - clip_start
                
                all_words.append({
                    "word": w["word"].strip().upper(),
                    "start": w_start,
                    "end": w_end
                })

    all_words.sort(key=lambda x: x["start"])

    log_info(f"[ass] Найдено слов для клипа: {len(all_words)}")

    if not all_words:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(header)
        return output_path

    for i in range(len(all_words)):
        if all_words[i]["end"] - all_words[i]["start"] < 0.25:
            all_words[i]["end"] = all_words[i]["start"] + 0.25
        
        if i < len(all_words) - 1:
            next_start = all_words[i+1]["start"]
            if all_words[i]["end"] > next_start:
                all_words[i]["end"] = next_start
            if next_start - all_words[i]["end"] > 0.5:
                all_words[i]["end"] = all_words[i]["start"] + 0.5

    group_size = 2
    for i in range(0, len(all_words), group_size):
        group = all_words[i:i + group_size]
        if not group:
            continue

        for j, active in enumerate(group):
            w_start = active["start"]
            w_end = active["end"]

            parts = []
            for k, w in enumerate(group):
                t = w["word"]
                if k == j:
                    parts.append(f"{{\\c&H00FFFF&\\fscx115\\fscy115}}{t}{{\\r}}")
                else:
                    parts.append(t)

            text = " ".join(parts)

            events.append(
                f"Dialogue: 0,{format_ass_time(w_start)},{format_ass_time(w_end)},"
                f"Default,,0,0,0,,{text}"
            )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(header)
        f.write("\n".join(events))
    
    log_info(f"[ass] Сгенерировано событий: {len(events)}")
    return output_path


def cut_clip(video_path, start, end, output_path, ass_path=None,
             mirror=False, speed_up=False):
    duration = end - start
    filters = []

    if mirror:
        filters.append("hflip")

    if speed_up:
        filters.append("setpts=0.95*PTS")

    filters.extend([
        "scale=1080:1920:force_original_aspect_ratio=decrease",
        "pad=1080:1920:(ow-iw)/2:(oh-ih)/2:color=black",
    ])

    if ass_path and os.path.exists(ass_path) and os.path.getsize(ass_path) > 100:
        simple_ass_path = f"C:\\temp\\{os.path.basename(ass_path)}"
        os.makedirs("C:\\temp", exist_ok=True)
        shutil.copy(ass_path, simple_ass_path)
        
        ass_abs = str(Path(simple_ass_path).resolve())
        ass_escaped = ass_abs.replace("\\", "/").replace(":", "\\:")
        filters.append(f"ass='{ass_escaped}'")
        log_info(f"[ffmpeg] Субтитры накладываются из {ass_escaped}")
    else:
        log_info(f"[ffmpeg] Субтитры НЕ накладываются (файл пуст или отсутствует)")

    cmd = [FFMPEG_EXE, "-ss", str(start), "-i", video_path, "-t", str(duration)]

    if speed_up:
        cmd += ["-filter:a", "atempo=1.05"]

    cmd += [
        "-vf", ",".join(filters),
        "-c:v", "libx264", 
        "-preset", "fast",
        "-crf", "23",
        "-g", "30",
        "-movflags", "+faststart",
        "-c:a", "aac",
        "-b:a", "128k",
        "-y",
        "-map_metadata", "-1",
        "-metadata", "encoder=CapCut",
        "-metadata", "software=CapCut",
        output_path,
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
        FFMPEG_EXE, "-i", video_path,
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


def run_pipeline_with_task(task_id: str, source_path: str, plan: str, user_id: str,
                            mirror: bool = False, speed_up: bool = False,
                            subtitles: bool = True):
    plan_config = PLANS.get(plan, PLANS["free"])
    max_clip_duration = plan_config["max_clip_duration"]
    use_watermark = plan_config["watermark"]
    max_clips = plan_config["max_clips"]

    video_path = source_path
    unique_prefix = str(int(time.time()))

    try:
        update_task(task_id, status="processing", progress=15, stage="Анализирую видео...")

        video_duration = get_video_duration(video_path)
        log_info(f"[task {task_id}] Длина видео: {video_duration:.1f}с")

        num_clips = calculate_num_clips(video_duration, max_clips)
        log_info(f"[task {task_id}] Клипов будет: {num_clips}")

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
            clip_filename = f"clip_{unique_prefix}_{i+1}.mp4"
            ass_filename = f"clip_{unique_prefix}_{i+1}.ass"
            thumb_filename = f"clip_{unique_prefix}_{i+1}.jpg"

            output = str(TEMP_DIR / clip_filename)
            ass_path = str(TEMP_DIR / ass_filename)
            thumb_path = str(TEMP_DIR / thumb_filename)

            if subtitles:
                generate_ass(segments, ass_path, moment["start"], moment["end"],
                             use_watermark=use_watermark)
            else:
                ass_path = None

            cut_clip(video_path, moment["start"], moment["end"], output, ass_path,
                     mirror=mirror, speed_up=speed_up)
            generate_thumb(output, thumb_path, at=0.5)

            clip_minutes = (moment["end"] - moment["start"]) / 60
            total_minutes += clip_minutes

            progress = 75 + int((i + 1) / len(moments) * 20)
            update_task(task_id, progress=progress, stage=f"Клип {i+1}/{len(moments)}")

            create_user_clip(
                user_id=user_id,
                task_id=task_id,
                title=moment["title"],
                start_time=moment["start"],
                end_time=moment["end"],
                filename=clip_filename,
                thumb_filename=thumb_filename if os.path.exists(thumb_path) else None,
            )

            clips.append({
                "id": i + 1,
                "title": moment["title"],
                "start": moment["start"],
                "end": moment["end"],
                "path": output,
                "thumb": thumb_path,
            })

        add_used_minutes(user_id, video_duration / 60)

        update_task(
            task_id,
            status="completed",
            progress=100,
            stage="Готово",
            clips=clips,
        )

        log_info(f"[task {task_id}] Завершено. Клипов: {len(clips)}")

    except PipelineError as e:
        log_error(f"[task {task_id}] PipelineError: {e}")
        update_task(task_id, status="error", error=str(e), stage="Ошибка")
    except Exception as e:
        log_error(f"[task {task_id}] Ошибка: {e}")
        update_task(task_id, status="error", error="Внутренняя ошибка", stage="Ошибка")
    finally:
        try:
            if os.path.exists(video_path) and str(UPLOAD_DIR) in video_path:
                os.remove(video_path)
                log_info(f"[task {task_id}] Исходник удалён")
        except Exception:
            pass