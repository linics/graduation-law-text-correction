from contextlib import contextmanager
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import schedule
import time
from threading import Thread
from sqlmodel import create_engine, Session, SQLModel, select

from models import Correction, Log, User

DB_PATH = "app.db"
engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})


def init_db():
    """Create tables and set WAL"""
    SQLModel.metadata.create_all(engine)
    with engine.connect() as conn:
        conn.execute("PRAGMA journal_mode=WAL;")

    schedule.every().day.at("03:00").do(archive_old)
    Thread(target=_run_scheduler, daemon=True).start()


@contextmanager
def get_session():
    with Session(engine) as session:
        yield session


def archive_old():
    cutoff = datetime.utcnow() - timedelta(days=90)
    Path("archive").mkdir(exist_ok=True)
    with Session(engine) as session:
        corrs = session.exec(select(Correction).where(Correction.ts < cutoff)).all()
        if corrs:
            df = pd.DataFrame([c.dict() for c in corrs])
            path = Path("archive") / f"corrections_{cutoff.date()}.csv.gz"
            df.to_csv(path, index=False, compression="gzip")
            for c in corrs:
                session.delete(c)
        logs = session.exec(select(Log).where(Log.ts < cutoff)).all()
        if logs:
            df = pd.DataFrame([l.dict() for l in logs])
            path = Path("archive") / f"logs_{cutoff.date()}.csv.gz"
            df.to_csv(path, index=False, compression="gzip")
            for l in logs:
                session.delete(l)
        session.commit()


def _run_scheduler():
    while True:
        schedule.run_pending()
        time.sleep(30)
