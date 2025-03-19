# 安装 Streamlit（若未安装）： pip install streamlit

import streamlit as st

st.title("简易 Web 应用示例")

# 文本输入组件
name = st.text_input("请输入你的名字：")

# 数字输入组件
number = st.number_input("选择一个数字：", min_value=0, max_value=100, value=50)

# 按钮触发事件
if st.button("提交"):
    # 在页面上显示结果
    st.write(f"你好，{name or '访客'}！你选择的数字的平方是 {number ** 2}。")
