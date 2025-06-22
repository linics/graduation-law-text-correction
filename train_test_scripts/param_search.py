import json
import itertools
import time
import Levenshtein
from the_graduation_design.text_correction_tool import TextCorrectionTool

def load_validation_data(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def compute_char_accuracy(pred, truth):
    max_len = max(len(pred), len(truth))
    pred = pred.ljust(max_len)
    truth = truth.ljust(max_len)
    return sum(p == t for p, t in zip(pred, truth)) / max_len

def evaluate_params(validation_data, tool, alpha, beta, gamma):
    total_edit_distance = 0
    exact_match_count = 0
    char_accuracy_total = 0
    top1_match_count = 0
    total_samples = 0

    for sample in validation_data:
        input_text = sample.get("input", "")
        true_output = sample.get("output", "")
        if not input_text or not true_output:
            continue

        pred_labels = tool.detect_errors_in_sentence(input_text)
        masked_text = tool.create_masked_text_from_predictions(input_text, pred_labels)

        corrected_text, _, _ = tool.iterative_correction(
            masked_text, input_text,
            max_iters=10,
            alpha=alpha, beta=beta, gamma=gamma,
            debug=True  # 打开 debug 模式
        )

        dist = Levenshtein.distance(corrected_text, true_output)
        total_edit_distance += dist
        total_samples += 1
        if corrected_text == true_output:
            exact_match_count += 1

        # 字符准确率
        char_accuracy_total += compute_char_accuracy(corrected_text, true_output)

        # 候选打分 top1 是否正确（由 tool 设置）
        if getattr(tool, "first_mask_top1_correct", False):
            top1_match_count += 1

    avg_edit_distance = total_edit_distance / total_samples if total_samples > 0 else float('inf')
    exact_match_rate = exact_match_count / total_samples if total_samples > 0 else 0
    avg_char_accuracy = char_accuracy_total / total_samples if total_samples > 0 else 0
    top1_match_rate = top1_match_count / total_samples if total_samples > 0 else 0
    return avg_edit_distance, exact_match_rate, avg_char_accuracy, top1_match_rate

if __name__ == "__main__":
    tool = TextCorrectionTool(
        detection_model_id="linics/detection-model",
        correction_model_id="linics/correction-model",
        device="cuda"
    )

    validation_file = "../dataset/law_correction_valid.jsonl"
    validation_data = load_validation_data(validation_file)
    print(f"加载验证集：{len(validation_data)} 条样本")

    alphas = [0.4, 0.5, 0.6, 0.7]
    betas = [0.1, 0.2, 0.3, 0.4]
    gammas = [0.1, 0.2, 0.3, 0.4]

    best_params = None
    best_metric = float('inf')
    results = []

    total_combinations = len(alphas) * len(betas) * len(gammas)
    print("开始参数搜索，共 {} 种组合".format(total_combinations))
    start_time = time.time()
    for alpha, beta, gamma in itertools.product(alphas, betas, gammas):
        print(f"评估参数组合：alpha={alpha}, beta={beta}, gamma={gamma} ...")
        avg_dist, exact_rate, char_acc, top1_acc = evaluate_params(validation_data, tool, alpha, beta, gamma)
        results.append((alpha, beta, gamma, avg_dist, exact_rate, char_acc, top1_acc))
        print(f"  平均编辑距离：{avg_dist:.2f}, 完全一致率：{exact_rate:.2%}, 字符准确率：{char_acc:.2%}, Top1匹配率：{top1_acc:.2%}")
        if avg_dist < best_metric:
            best_metric = avg_dist
            best_params = (alpha, beta, gamma)
    duration = time.time() - start_time
    print("\\n参数搜索完成，耗时 {:.2f} 秒".format(duration))
    print("最佳参数组合：alpha={}, beta={}, gamma={}".format(*best_params))
    print("对应平均编辑距离：{:.2f}".format(best_metric))

    try:
        import pandas as pd
        df = pd.DataFrame(results, columns=["alpha", "beta", "gamma", "avg_edit_distance", "exact_match_rate", "char_accuracy", "top1_match_rate"])
        df.to_csv("parameter_search_results.csv", index=False)
        print("参数搜索结果已保存到 parameter_search_results.csv")
    except Exception as e:
        print("保存结果失败：", e)