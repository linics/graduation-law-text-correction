import streamlit as st

from services.kv_config import get_all, save
from services.auth import log_action

st.title("系统设置")

cfg = get_all()

for k, v in cfg.items():
    if st.session_state.get('role') == 'admin':
        new_v = st.number_input(k, value=float(v), min_value=0.0, max_value=1.0, step=0.05)
        if new_v != float(v):
            save(k, str(new_v))
            st.toast("已保存")
    else:
        st.number_input(k, value=float(v), disabled=True)

log_action(st.session_state.get('user',''), 'view_settings', '')
