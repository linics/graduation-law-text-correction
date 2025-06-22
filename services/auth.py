from datetime import datetime, timedelta, date
from passlib.hash import pbkdf2_sha256
from sqlalchemy import text
from sqlmodel import select

from .db import get_session
from models import User, Log, KVConfig


def register(username: str, password: str, role: str = "user") -> bool:
    """Register a new user, return False if username exists"""
    with get_session() as session:
        if session.exec(select(User).where(User.username == username)).first():
            return False
        user = User(username=username, pwd_hash=pbkdf2_sha256.hash(password), role=role)
        session.add(user)
        session.commit()
        return True


def verify(username: str, password: str) -> tuple[bool, str]:
    """Verify user credentials with lock mechanism"""
    with get_session() as session:
        user = session.exec(select(User).where(User.username == username)).first()
        now = datetime.utcnow()
        if not user:
            return False, ""
        if user.lock_until and user.lock_until > now:
            return False, ""
        if pbkdf2_sha256.verify(password, user.pwd_hash):
            user.failed_attempts = 0
            session.add(user)
            session.commit()
            return True, user.role
        user.failed_attempts += 1
        if user.failed_attempts >= 5:
            user.lock_until = now + timedelta(minutes=10)
            user.failed_attempts = 0
        session.add(user)
        session.commit()
        return False, ""


def log_action(user: str, action: str, payload: str = ""):
    with get_session() as session:
        session.add(Log(user=user, action=action, payload=payload))
        session.commit()


def get_daily_limit() -> int:
    with get_session() as session:
        cfg = session.exec(select(KVConfig).where(KVConfig.key == "daily_limit")).first()
        return int(cfg.value) if cfg else 20


def check_and_inc_quota(username: str) -> bool:
    limit = get_daily_limit()
    today = str(date.today())
    with get_session() as session:
        day_flag = session.exec(select(KVConfig).where(KVConfig.key == "quota_day")).first()
        if not day_flag or day_flag.value != today:
            session.exec(select(User))  # open transaction
            session.execute(text("UPDATE user SET request_today=0"))
            if day_flag:
                day_flag.value = today
                session.add(day_flag)
            else:
                session.add(KVConfig(key="quota_day", value=today))
            session.commit()
        user = session.exec(select(User).where(User.username == username)).first()
        if not user:
            return False
        if user.request_today >= limit:
            session.add(Log(user=username, action="quota_exceed", payload=""))
            session.commit()
            return False
        user.request_today += 1
        session.add(user)
        session.commit()
        return True
