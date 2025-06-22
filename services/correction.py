from difflib import ndiff
import json
from sqlmodel import select
from models import Correction
from .db import get_session


def calc_diff(a: str, b: str) -> str:
    diff = list(ndiff(a, b))
    return json.dumps(diff, ensure_ascii=False)


def save_correction(user: str, raw: str, corrected: str, model_ver: str):
    with get_session() as session:
        diff_json = calc_diff(raw, corrected)
        session.add(Correction(user=user, raw_text=raw, corrected_text=corrected,
                               diff_json=diff_json, model_ver=model_ver))
        session.commit()


def load_history(username: str | None, start: str | None, end: str | None, offset: int, limit: int):
    with get_session() as session:
        stmt = select(Correction)
        if username:
            stmt = stmt.where(Correction.user == username)
        if start:
            stmt = stmt.where(Correction.ts >= start)
        if end:
            stmt = stmt.where(Correction.ts <= end)
        stmt = stmt.order_by(Correction.ts.desc()).offset(offset).limit(limit)
        rows = session.exec(stmt).all()
        total = session.exec(select(Correction).count()).one()
        return rows, total
