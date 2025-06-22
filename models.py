from datetime import datetime
from sqlmodel import SQLModel, Field, Column, TEXT, Index


class User(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    username: str = Field(sa_column=Column("username", TEXT, unique=True))
    pwd_hash: str
    role: str = "user"
    failed_attempts: int = 0
    lock_until: datetime | None = None
    request_today: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Correction(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    user: str
    raw_text: str
    corrected_text: str
    diff_json: str
    model_ver: str
    ts: datetime = Field(default_factory=datetime.utcnow, index=True)

    __table_args__ = (Index("idx_corr_user_ts", "user", "ts"),)


class KVConfig(SQLModel, table=True):
    key: str = Field(primary_key=True)
    value: str


class Log(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    user: str
    action: str
    payload: str
    ts: datetime = Field(default_factory=datetime.utcnow, index=True)

    __table_args__ = (Index("idx_log_ts", "ts"),)
