import streamlit as st
import Levenshtein
import io
import re
from difflib import SequenceMatcher
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import time

from text_correction_tool import TextCorrectionTool


# ===============================
# 1. 工具函数
# ===============================
def highlight_diff(original, corrected):
    """
    使用 difflib.SequenceMatcher 对比原句与纠错后句，返回 HTML 字符串。
    - 删除的内容用红底+删除线显示
    - 新增的内容用绿底显示
    """
    matcher = SequenceMatcher(None, original, corrected)
    result = []
    for op, a1, a2, b1, b2 in matcher.get_opcodes():
        if op == 'equal':
            result.append(original[a1:a2])
        elif op == 'insert':
            inserted = corrected[b1:b2]
            result.append(f"<span style='background-color:#d4f7d4;'>{inserted}</span>")
        elif op == 'delete':
            deleted = original[a1:a2]
            result.append(f"<span style='background-color:#ffdce0;text-decoration:line-through;'>{deleted}</span>")
        elif op == 'replace':
            replaced_orig = original[a1:a2]
            replaced_new  = corrected[b1:b2]
            result.append(f"<span style='background-color:#ffdce0;text-decoration:line-through;'>{replaced_orig}</span>")
            result.append(f"<span style='background-color:#d4f7d4;'>{replaced_new}</span>")
    return "".join(result)

def split_sentences(text):
    """
    按照 。！？ 进行分句，返回一个句子列表。
    若用户上传了整段文本，则自动拆分为若干句子。
    """
    # 替换全角问号、感叹号，防止切分异常
    # 如果文本中包含英文句号等，可以根据需求再做处理
    # 这里用 re.split 来拆分，并保留分割符以便在处理时可选
    sentences = re.split(r'[。！？]', text)
    # 去除空白
    return [s.strip() for s in sentences if s.strip()]

# ===============================
# 2. 缓存加载模型
# ===============================
@st.cache_resource(show_spinner=True)
def load_tool():
    # 根据你上传到 Hugging Face 的仓库 ID 或本地路径来加载
    return TextCorrectionTool(
        detection_model_id="linics/detection-model",      # 你的检测模型ID
        correction_model_id="linics/correction-model",    # 你的纠错模型ID
        device="gpu"  # 如果云端没有GPU，可以使用 "cpu"
    )

# 在脚本最前面调用 set_page_config
st.set_page_config(page_title="中文文本纠错系统", layout="wide")

tool = load_tool()

# ===============================
# 3. 页面布局
# ===============================
st.title("中文法律文本纠错系统")
st.write("本系统基于错误检测与纠错模型，实现了文本纠错功能。支持单句与批量文本处理。")

# 在侧边栏放置纠错参数
st.sidebar.header("纠错参数设置")
alpha = st.sidebar.slider("Alpha (模型得分权重)", 0.0, 1.0, 0.4, 0.05)
beta  = st.sidebar.slider("Beta (拼音相似度权重)", 0.0, 1.0, 0.4, 0.05)
gamma = st.sidebar.slider("Gamma (字形相似度权重)", 0.0, 1.0, 0.1, 0.05)
debug_mode = st.sidebar.checkbox("显示候选词细节", value=False)
st.sidebar.info("调节纠错时模型得分、拼音相似度、字形相似度的权重；\n勾选 '显示候选词细节' 以查看每次迭代替换时的候选词。")

tab_single, tab_batch = st.tabs(["单句纠错", "批量纠错"])

# ===============================
# 4. 单句纠错
# ===============================
with tab_single:
    input_text = st.text_area("请输入待纠错的中文句子：", height=120)
    if st.button("开始纠错", key="single_correct"):
        if not input_text.strip():
            st.warning("请输入文本内容。")
        else:
            with st.spinner("纠错中，请稍候..."):
                start_time = time.time()
                # 1. 检测
                pred_labels = tool.detect_errors_in_sentence(input_text)
                # 2. 生成 Mask
                masked_text = tool.create_masked_text_from_predictions(input_text, pred_labels)
                # 3. 迭代纠错
                result = tool.iterative_correction(
                    masked_text, input_text,
                    max_iters=10,
                    alpha=alpha, beta=beta, gamma=gamma,
                    debug=debug_mode
                )
                if debug_mode:
                    corrected_text, predictions_list = result
                else:
                    corrected_text = result
                duration = time.time() - start_time

            st.subheader("纠错结果")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**原始文本：**")
                st.write(input_text)
            with col2:
                st.markdown("**纠错后文本：**")
                st.write(corrected_text)
            st.write(f"处理时间：{duration:.2f} 秒")

            diff_html = highlight_diff(input_text, corrected_text)
            st.markdown("**差异高亮**")
            st.markdown(f"<div style='font-size:1.1rem;'>{diff_html}</div>", unsafe_allow_html=True)

            edit_dist = Levenshtein.distance(corrected_text, input_text)
            st.write(f"**编辑距离：** {edit_dist}")

            if debug_mode:
                with st.expander("查看候选词细节"):
                    for i, candidates in enumerate(predictions_list):
                        st.markdown(f"**第 {i+1} 次替换候选词**")
                        top5 = candidates[:5]
                        for cand in top5:
                            token_str, combined, model_score, pinyin_sim, shape_sim = cand
                            st.write(f"- {token_str} | 综合: {combined:.3f}, 模型: {model_score:.3f}, 拼音: {pinyin_sim:.3f}, 字形: {shape_sim:.3f}")

# ===============================
# 5. 批量纠错
# ===============================
with tab_batch:
    st.write("批量纠错：系统会自动按 '。！？' 分句。")
    uploaded_file = st.file_uploader("上传一个 .txt 文件，内容可包含多行或段落", type=["txt"])
    if uploaded_file is not None:
        content = uploaded_file.read().decode("utf-8").strip()
        # 将整个文件内容进行按行拆分，并再对每行进行分句
        lines = content.splitlines()
        sentences = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # 按 '。！？' 分割成句子
            splitted = split_sentences(line)
            sentences.extend(splitted)

        if st.button("开始批量纠错", key="batch_correct"):
            if not sentences:
                st.warning("文件中没有有效句子。")
            else:
                total = len(sentences)
                progress_bar = st.progress(0)
                results = []
                edit_dists = []
                start_time = time.time()
                for idx, sent in enumerate(sentences, start=1):
                    pred_labels = tool.detect_errors_in_sentence(sent)
                    masked_text = tool.create_masked_text_from_predictions(sent, pred_labels)
                    corrected_line = tool.iterative_correction(
                        masked_text, sent,
                        max_iters=10,
                        alpha=alpha, beta=beta, gamma=gamma,
                        debug=False
                    )
                    dist = Levenshtein.distance(corrected_line, sent)
                    results.append((sent, corrected_line, dist))
                    edit_dists.append(dist)
                    progress_bar.progress(idx/total)
                progress_bar.empty()
                duration = time.time() - start_time

                st.subheader("批量纠错结果")
                st.write(f"处理总句数：{len(results)}，耗时：{duration:.2f} 秒")
                for idx, (orig, corr, dist) in enumerate(results, start=1):
                    st.markdown(f"**句子 {idx}**")
                    diff_html = highlight_diff(orig, corr)
                    st.markdown(f"<div style='font-size:1.0rem;'>{diff_html}</div>", unsafe_allow_html=True)
                    st.write(f"编辑距离：{dist}")
                    st.write("---")

                # 绘制编辑距离直方图
                st.subheader("编辑距离分布")
                # 设置中文字体
                matplotlib.rc("font", family="SimHei")
                fig, ax = plt.subplots(figsize=(5, 3))  # 调整图形尺寸
                ax.hist(edit_dists, bins=10, color="#69b3a2", edgecolor="black")
                ax.set_xlabel("编辑距离")
                ax.set_ylabel("句子数量")
                st.pyplot(fig, use_container_width=False)

