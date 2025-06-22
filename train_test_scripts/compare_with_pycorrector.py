import sys
import json
import Levenshtein
import pandas as pd

# 加入 pycorrector 路径（请根据你自己的实际路径修改）
sys.path.append("D:/deeplearning/pycorrector-master")

from pycorrector.corrector import Corrector
from pycorrector.macbert.macbert_corrector import MacBertCorrector

# 初始化模型
default_corrector = Corrector()
macbert = MacBertCorrector()

# 加载测试集（路径来自 total_test.py）
test_file_path = r"D:\deeplearning\the_graduation_design\dataset\law_correction_test.jsonl"
with open(test_file_path, "r", encoding="utf-8") as f:
    test_data = [json.loads(line.strip()) for line in f if line.strip()]

# 统一评估函数
def evaluate_model(correct_func, name):
    total = len(test_data)
    exact_match_count = 0
    total_edit_distance = 0
    total_char_accuracy = 0

    for item in test_data:
        input_text = item["input"]
        target_text = item["output"]

        try:
            result = correct_func(input_text)
            if isinstance(result, dict):
                corrected_text = result.get("target", input_text)
            elif isinstance(result, tuple):
                corrected_text = result[0]
            else:
                corrected_text = result
        except Exception:
            corrected_text = input_text

        if corrected_text == target_text:
            exact_match_count += 1

        dist = Levenshtein.distance(corrected_text, target_text)
        total_edit_distance += dist

        max_len = max(len(corrected_text), len(target_text))
        correct_count = sum(1 for a, b in zip(corrected_text.ljust(max_len), target_text.ljust(max_len)) if a == b)
        total_char_accuracy += correct_count / max_len

    return {
        "系统": name,
        "完全匹配率 (%)": round(exact_match_count / total * 100, 2),
        "平均编辑距离": round(total_edit_distance / total, 2),
        "字符准确率 (%)": round(total_char_accuracy / total * 100, 2)
    }


# 执行两个 pycorrector 模型评估
results = [
    evaluate_model(default_corrector.correct, "pycorrector-规则模型"),
    evaluate_model(macbert.correct, "pycorrector-MacBERT模型")
]

# 输出表格
df = pd.DataFrame(results)
print(df.to_string(index=False))  # 打印表格
df.to_csv("pycorrector_eval_result.csv", index=False, encoding="utf-8-sig")  # 保存


from pycorrector.corrector import Corrector

corrector = Corrector()

test_sentences = [
    "我喜欢吃苹果，但苹果很回。",    # “回”应为“甜”
    "这个行为是违法德。",            # “德”应为“的”
    "人民法院神力行政案件时，应当对下列哪项进行审查:",  # “神力”应为“审理”
]

print("\n【规则模型测试样例】")
for s in test_sentences:
    corrected, details = corrector.correct(s)
    print(f"\n原句：{s}")
    print(f"纠错：{corrected}")
    print(f"详情：{details}")
