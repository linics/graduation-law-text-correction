import pandas as pd
import streamlit as st

from services.auth import log_action
from services.db import get_session
from models import Log
from sqlmodel import select
from sqlalchemy import func

st.title("日志管理")

page = st.number_input("页码", min_value=1, step=1, value=1)
page_size = 20
with get_session() as session:
    total = session.exec(select(func.count()).select_from(Log)).one()
    logs = session.exec(select(Log).order_by(Log.id.desc()).offset((page-1)*page_size).limit(page_size)).all()

df = pd.DataFrame([l.dict() for l in logs])

st.dataframe(df)

csv = df.to_csv(index=False).encode("utf-8")
st.download_button("导出当前页", csv, "logs_page.csv")

log_action(st.session_state.get('user',''), 'view_logs', str(page))
