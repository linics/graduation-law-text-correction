from sqlmodel import select
from .db import get_session
from models import KVConfig


def get_all() -> dict:
    with get_session() as session:
        pairs = session.exec(select(KVConfig)).all()
        return {p.key: p.value for p in pairs}


def save(key: str, value: str):
    with get_session() as session:
        cfg = session.exec(select(KVConfig).where(KVConfig.key == key)).first()
        if cfg:
            cfg.value = value
            session.add(cfg)
        else:
            session.add(KVConfig(key=key, value=value))
        session.commit()
