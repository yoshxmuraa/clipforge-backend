import os
import json
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")


def get_conn():
    return psycopg2.connect(DATABASE_URL)


def get_profile(user_id: str):
    try:
        conn = get_conn()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("select * from profiles where id = %s", (user_id,))
        row = cur.fetchone()
        cur.close()
        conn.close()
        return dict(row) if row else None
    except Exception as e:
        print(f"[db] get_profile error: {e}")
        return None


def get_profile_by_email(email: str):
    try:
        conn = get_conn()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("select * from profiles where email = %s", (email,))
        row = cur.fetchone()
        cur.close()
        conn.close()
        return dict(row) if row else None
    except Exception as e:
        print(f"[db] get_profile_by_email error: {e}")
        return None


def create_profile(email: str, password_hash: str, name: str = None):
    try:
        conn = get_conn()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            "insert into profiles (email, password_hash, name) values (%s, %s, %s) returning *",
            (email, password_hash, name),
        )
        row = cur.fetchone()
        conn.commit()
        cur.close()
        conn.close()
        return dict(row) if row else None
    except Exception as e:
        print(f"[db] create_profile error: {e}")
        return None


def update_profile(user_id: str, **fields):
    if not fields:
        return None
    try:
        conn = get_conn()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        set_clause = ", ".join([f"{k} = %s" for k in fields.keys()])
        values = list(fields.values()) + [user_id]
        cur.execute(f"update profiles set {set_clause} where id = %s returning *", values)
        row = cur.fetchone()
        conn.commit()
        cur.close()
        conn.close()
        return dict(row) if row else None
    except Exception as e:
        print(f"[db] update_profile error: {e}")
        return None


def add_used_minutes(user_id: str, minutes: float):
    profile = get_profile(user_id)
    if not profile:
        return
    new_used = (profile.get("used_minutes") or 0) + minutes
    update_profile(user_id, used_minutes=new_used)


def create_task(user_id: str, url: str, plan: str):
    try:
        conn = get_conn()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            "insert into tasks (user_id, url, plan, status, progress, stage) "
            "values (%s, %s, %s, 'pending', 0, 'Ожидание') returning *",
            (user_id, url, plan),
        )
        row = cur.fetchone()
        conn.commit()
        cur.close()
        conn.close()
        return dict(row) if row else None
    except Exception as e:
        print(f"[db] create_task error: {e}")
        return None


def update_task(task_id: str, **fields):
    if not fields:
        return None
    try:
        conn = get_conn()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        set_parts = []
        values = []
        for k, v in fields.items():
            if k == "clips" and isinstance(v, (list, dict)):
                set_parts.append(f"{k} = %s::jsonb")
                values.append(json.dumps(v))
            else:
                set_parts.append(f"{k} = %s")
                values.append(v)
        set_clause = ", ".join(set_parts)
        values.append(task_id)
        cur.execute(f"update tasks set {set_clause}, updated_at = now() where id = %s returning *", values)
        row = cur.fetchone()
        conn.commit()
        cur.close()
        conn.close()
        return dict(row) if row else None
    except Exception as e:
        print(f"[db] update_task error: {e}")
        return None


def get_task(task_id: str):
    try:
        conn = get_conn()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("select * from tasks where id = %s", (task_id,))
        row = cur.fetchone()
        cur.close()
        conn.close()
        return dict(row) if row else None
    except Exception as e:
        print(f"[db] get_task error: {e}")
        return None


def get_active_task(user_id: str):
    try:
        conn = get_conn()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            "select * from tasks where user_id = %s and status in ('pending', 'processing') "
            "order by created_at desc limit 1",
            (user_id,),
        )
        row = cur.fetchone()
        cur.close()
        conn.close()
        return dict(row) if row else None
    except Exception as e:
        print(f"[db] get_active_task error: {e}")
        return None


def create_user_clip(user_id: str, task_id: str, title: str,
                     start_time: float, end_time: float,
                     filename: str, thumb_filename: str = None):
    try:
        conn = get_conn()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            "insert into user_clips (user_id, task_id, title, start_time, end_time, filename, thumb_filename) "
            "values (%s, %s, %s, %s, %s, %s, %s) returning *",
            (user_id, task_id, title, start_time, end_time, filename, thumb_filename),
        )
        row = cur.fetchone()
        conn.commit()
        cur.close()
        conn.close()
        return dict(row) if row else None
    except Exception as e:
        print(f"[db] create_user_clip error: {e}")
        return None


def get_user_clips(user_id: str):
    try:
        conn = get_conn()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            "select * from user_clips where user_id = %s order by created_at desc",
            (user_id,),
        )
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        print(f"[db] get_user_clips error: {e}")
        return []


def delete_user_clips(user_id: str, clip_ids: list):
    if not clip_ids:
        return []
    try:
        conn = get_conn()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            "select filename, thumb_filename from user_clips where user_id = %s and id = any(%s)",
            (user_id, clip_ids),
        )
        rows = cur.fetchall()
        cur.execute(
            "delete from user_clips where user_id = %s and id = any(%s)",
            (user_id, clip_ids),
        )
        conn.commit()
        cur.close()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        print(f"[db] delete_user_clips error: {e}")
        return []