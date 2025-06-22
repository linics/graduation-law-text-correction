import pandas as pd
import streamlit as st
from datetime import date

from services.correction import load_history


st.title("历史纠错记录")

user = st.text_input("筛选用户")
col1, col2 = st.columns(2)
with col1:
    start = st.date_input("起始日期", value=None)
with col2:
    end = st.date_input("结束日期", value=None)
page = st.number_input("页码", min_value=1, step=1, value=1)
page_size = 20
records, total = load_history(user or None, str(start) if start else None,
                              str(end) if end else None,
                              (page-1)*page_size, page_size)

df = pd.DataFrame([r.dict() for r in records])
selected = st.radio("选择要加载的记录", options=df.index if not df.empty else [], format_func=lambda x: df.loc[x, 'raw_text'] if not df.empty else '')

st.dataframe(df)

if st.button("加载到主页面") and not df.empty:
    st.session_state['input_text'] = df.loc[selected, 'raw_text']
    st.switch_page("app.py")
