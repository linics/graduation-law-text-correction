import streamlit as st
import Levenshtein
import io
from difflib import SequenceMatcher

from text_correction_tool import TextCorrectionTool

# 设置页面配置
st.set_page_config(page_title="中文文本纠错系统", layout="wide")


# ===============================
# 1. 工具函数：高亮原文与纠错后句的差异
# ===============================
def highlight_diff(original, corrected):
    """
    使用 difflib.SequenceMatcher 对比原句与纠错后句，返回 HTML 字符串。
    - 删除的内容用红底+删除线
    - 新增的内容用绿底
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
            replaced_new = corrected[b1:b2]
            result.append(
                f"<span style='background-color:#ffdce0;text-decoration:line-through;'>{replaced_orig}</span>")
            result.append(f"<span style='background-color:#d4f7d4;'>{replaced_new}</span>")
    return "".join(result)


# ===============================
# 2. 缓存加载模型
# ===============================
@st.cache_resource(show_spinner=True)
def load_tool():
    return TextCorrectionTool(
        detection_model_id="linics/detection-model",
        correction_model_id="linics/correction-model",
        device="cuda"
    )


tool = load_tool()

# ===============================
# 3. Streamlit 页面布局
# ===============================
st.title("中文文本纠错系统")

# 在左侧栏放置纠错参数
st.sidebar.header("纠错参数设置")
alpha = st.sidebar.slider("Alpha (模型得分权重)", 0.0, 1.0, 0.7, 0.05)
beta = st.sidebar.slider("Beta (拼音相似度权重)", 0.0, 1.0, 0.3, 0.05)
gamma = st.sidebar.slider("Gamma (字形相似度权重)", 0.0, 1.0, 0.3, 0.05)
debug_mode = st.sidebar.checkbox("显示候选词细节", value=False)
st.sidebar.info("可调节模型分、拼音和字形相似度在纠错过程中的占比。")

# ===============================
# 4. 输入方式：单句 or 批量上传
# ===============================
st.write("你可以输入一句文本进行纠错，或者上传一个 .txt 文件进行批量纠错。")

tab_single, tab_batch = st.tabs(["单句纠错", "批量纠错"])

with tab_single:
    input_text = st.text_area("请输入待纠错的中文句子：", height=120)
    if st.button("开始纠错", key="single_correct"):
        if not input_text.strip():
            st.warning("请输入文本内容。")
        else:
            # 1. 使用检测模型得到错误标签，并生成 mask 文本
            pred_labels = tool.detect_errors_in_sentence(input_text)
            masked_text = tool.create_masked_text_from_predictions(input_text, pred_labels)

            # 2. 迭代纠错，根据 debug_mode 分别处理返回值
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

            # 显示结果
            st.subheader("结果展示")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**原始文本：**")
                st.write(input_text)
            with col2:
                st.markdown("**纠错后文本：**")
                st.write(corrected_text)

            # 高亮对比
            st.markdown("**差异高亮**")
            diff_html = highlight_diff(input_text, corrected_text)
            st.markdown(f"<div style='font-size:1.1rem;'>{diff_html}</div>", unsafe_allow_html=True)

            # 显示编辑距离
            edit_dist = Levenshtein.distance(corrected_text, input_text)
            st.write(f"**编辑距离：** {edit_dist}")

            # 若 debug_mode=True，显示候选词细节
            if debug_mode and predictions_list:
                with st.expander("查看候选词细节"):
                    for i, candidates in enumerate(predictions_list):
                        st.markdown(f"**第 {i + 1} 次替换**")
                        top5 = candidates[:5]  # 显示前 5 个候选词
                        for cand in top5:
                            token_str, combined, model_score, pinyin_sim, shape_sim = cand
                            st.write(
                                f"- {token_str} | 综合: {combined:.3f}, 模型: {model_score:.3f}, 拼音: {pinyin_sim:.3f}, 字形: {shape_sim:.3f}")

with tab_batch:
    uploaded_file = st.file_uploader("上传一个 .txt 文件，每行一句", type=["txt"])
    if uploaded_file is not None:
        content = uploaded_file.read().decode("utf-8").strip().splitlines()
        if st.button("开始批量纠错", key="batch_correct"):
            results = []
            for idx, line in enumerate(content, start=1):
                if not line.strip():
                    continue
                pred_labels = tool.detect_errors_in_sentence(line)
                masked_text = tool.create_masked_text_from_predictions(line, pred_labels)
                corrected_line = tool.iterative_correction(
                    masked_text, line,
                    max_iters=10,
                    alpha=alpha, beta=beta, gamma=gamma,
                    debug=False
                )
                results.append((line, corrected_line))
            st.subheader("批量纠错结果")
            for idx, (orig, corr) in enumerate(results, start=1):
                st.markdown(f"**句子 {idx}**")
                diff_html = highlight_diff(orig, corr)
                st.markdown(f"<div style='font-size:1.0rem;'>{diff_html}</div>", unsafe_allow_html=True)
                st.write("---")
