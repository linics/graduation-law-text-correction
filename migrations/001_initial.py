"""Initial database creation"""
from sqlmodel import SQLModel
from models import User, Correction, KVConfig, Log
from services.db import engine


def run():
    SQLModel.metadata.create_all(engine)


if __name__ == "__main__":
    run()
    print("database initialized")
