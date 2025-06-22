"""Initial database creation"""
from sqlmodel import SQLModel, Session, select
from models import User, Correction, KVConfig, Log
from services.db import engine


def run():
    SQLModel.metadata.create_all(engine)
    defaults = {
        "alpha": "0.4",
        "beta": "0.4",
        "gamma": "0.1",
        "daily_limit": "20",
    }
    with Session(engine) as session:
        for k, v in defaults.items():
            exists = session.exec(select(KVConfig).where(KVConfig.key == k)).first()
            if not exists:
                session.add(KVConfig(key=k, value=v))
        session.commit()


if __name__ == "__main__":
    run()
    print("database initialized")
