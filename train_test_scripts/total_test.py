import os
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    pipeline,
    DataCollatorForTokenClassification,
    Trainer
)
import Levenshtein  # pip install python-Levenshtein
import pypinyin
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


#############################################
# 1. 错误检测部分

def compute_token_classification_metrics(predictions, labels):
    """
    计算 token 级别的各项指标（忽略 -100 标签位置）。
    """
    predictions = np.argmax(predictions, axis=2)
    true_preds = []
    true_labels = []
    for pred_seq, label_seq in zip(predictions, labels):
        for p, l in zip(pred_seq, label_seq):
            if l != -100:
                true_preds.append(p)
                true_labels.append(l)
    accuracy = accuracy_score(true_labels, true_preds)
    precision = precision_score(true_labels, true_preds, average='binary')
    recall = recall_score(true_labels, true_preds, average='binary')
    f1 = f1_score(true_labels, true_preds, average='binary')
    report = classification_report(true_labels, true_preds, target_names=["correct", "error"])
    return accuracy, precision, recall, f1, report


def find_error_positions(input_text, output_text):
    """
    按字符比较 input_text 与 output_text，不同位置标记为 1（错误），相同为 0。
    """
    min_length = min(len(input_text), len(output_text))
    error_positions = [1 if input_text[i] != output_text[i] else 0 for i in range(min_length)]
    if len(input_text) > min_length:
        error_positions.extend([0] * (len(input_text) - min_length))
    return error_positions


def tokenize_and_align_labels(examples):
    """
    对样本中的 "input" 进行按字符拆分，并依据真实输出计算标签（0正确，1错误）。
    利用 tokenizer 对字符进行处理，并对齐标签（非首个 sub-token 标记为 -100）。
    """
    inputs_split = [list(s) for s in examples["input"]]
    tokenized_inputs = detection_tokenizer(
        inputs_split,
        is_split_into_words=True,
        padding="max_length",
        truncation=True,
        max_length=128,
    )
    labels = []
    for i in range(len(examples["input"])):
        input_text = examples["input"][i]
        output_text = examples["output"][i]
        if examples["error_count"][i] == 0:
            label_ids = [0] * len(input_text)
        else:
            label_ids = find_error_positions(input_text, output_text)
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        aligned_labels = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                aligned_labels.append(-100)
            elif word_idx != previous_word_idx:
                aligned_labels.append(label_ids[word_idx])
            else:
                aligned_labels.append(-100)
            previous_word_idx = word_idx
        labels.append(aligned_labels)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def detect_errors_in_sentence(sentence, model, tokenizer, max_length=128):
    """
    对单个句子进行错误检测，返回每个字符的预测标签列表（0表示正确，1表示错误）。
    先获取原始 Tokenizer 结果以提取 word_ids，然后将张量转移到模型所在设备。
    """
    inputs = list(sentence)
    raw_tokenized = tokenizer(
        [inputs],
        is_split_into_words=True,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    word_ids = raw_tokenized.word_ids(batch_index=0)  # 从原始结果中获取 word_ids

    device = next(model.parameters()).device
    tokenized_input = {k: v.to(device) for k, v in raw_tokenized.items()}

    outputs = model(**tokenized_input)
    predictions = outputs.logits.detach().cpu().numpy()
    predictions = np.argmax(predictions, axis=2)[0]

    pred_labels = []
    previous_word_idx = None
    for word_idx, pred in zip(word_ids, predictions):
        if word_idx is None:
            continue
        elif word_idx != previous_word_idx:
            pred_labels.append(pred)
        previous_word_idx = word_idx
    return pred_labels


def create_masked_text_from_predictions(sentence, pred_labels):
    """
    根据句子及每个字符的预测标签生成 mask 句：
    预测为错误的位置替换为 "[MASK]"，其它位置保留原字符。
    要求 len(sentence) == len(pred_labels)。
    """
    masked_chars = []
    for char, label in zip(sentence, pred_labels):
        if label == 1:
            masked_chars.append("[MASK]")
        else:
            masked_chars.append(char)
    return "".join(masked_chars)


#############################################
# 2. 纠错部分

def get_pinyin(text):
    """将文本转换为拼音字符串"""
    return "".join(pypinyin.lazy_pinyin(text))


def compute_pinyin_similarity(candidate, target):
    """
    计算候选词与目标词的拼音相似度：
    拼音完全一致返回 1.0，否则根据编辑距离归一化相似度。
    """
    cand_pinyin = get_pinyin(candidate)
    target_pinyin = get_pinyin(target)
    if cand_pinyin == target_pinyin:
        return 1.0
    max_len = max(len(cand_pinyin), len(target_pinyin))
    sim = 1 - Levenshtein.distance(cand_pinyin, target_pinyin) / max_len
    return sim


def compute_shape_similarity(candidate, target):
    """
    计算候选词与目标词的字形相似度：
    若候选词与目标词完全一致返回 1.0，否则返回 0.5（示例）。
    """
    return 1.0 if candidate == target else 0.5


def get_all_candidate_scores(candidate_list, target, alpha=0.7, beta=0.3, gamma=0.3):
    """
    计算候选列表中每个候选词的综合得分：
      综合得分 = alpha * 模型得分 + beta * 拼音相似度 + gamma * 字形相似度。
    返回排序后的列表，每个元素为 (候选词, 综合得分, 模型得分, 拼音相似度, 字形相似度)。
    """
    results = []
    for cand in candidate_list:
        model_score = cand["score"]
        pinyin_sim = compute_pinyin_similarity(cand["token_str"], target)
        shape_sim = compute_shape_similarity(cand["token_str"], target)
        combined = alpha * model_score + beta * pinyin_sim + gamma * shape_sim
        results.append((cand["token_str"], combined, model_score, pinyin_sim, shape_sim))
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def iterative_correction(corrector, masked_text, target_text, max_iters=10, alpha=0.7, beta=0.3, gamma=0.3):
    """
    对 masked_text 进行迭代纠错，直至句中不再含 "[MASK]" 或达到最大迭代次数。
    target_text 用于计算候选词的拼音和字形相似度。
    返回最终纠错句及每次迭代候选词的评分记录。
    """
    corrected_sentence = masked_text
    predictions_list = []
    iters = 0
    while "[MASK]" in corrected_sentence and iters < max_iters:
        try:
            results = corrector(corrected_sentence)
            candidate_list = results[0] if isinstance(results[0], list) else results
            mask_index = corrected_sentence.index("[MASK]")
            if mask_index < len(target_text):
                target_char = target_text[mask_index]
            else:
                target_char = ""
            candidates_with_scores = get_all_candidate_scores(candidate_list, target_char, alpha=alpha, beta=beta,
                                                              gamma=gamma)
            predictions_list.append(candidates_with_scores)
            top_candidate = candidates_with_scores[0][0]
            corrected_sentence = corrected_sentence.replace("[MASK]", top_candidate, 1)
        except Exception as e:
            print("fill-mask 出错:", e)
            break
        iters += 1
    return corrected_sentence, predictions_list


def create_soft_masked_data(example):
    """
    根据样本的 input 与 output 比较生成 masked_input：
    若对应字符不同，则替换为 "[MASK]"，否则保留原字符（完美检测）。
    """
    input_text = example["input"]
    output_text = example["output"]
    if example.get("error_count", 0) == 0:
        return {"masked_input": input_text}
    tokens = []
    min_len = min(len(input_text), len(output_text))
    for i in range(min_len):
        if input_text[i] != output_text[i]:
            tokens.append("[MASK]")
        else:
            tokens.append(input_text[i])
    if len(input_text) > min_len:
        tokens.extend(list(input_text[min_len:]))
    return {"masked_input": "".join(tokens)}


#############################################
# 3. 三种评估方式（保留）

def evaluate_detection(detection_model, detection_tokenizer, test_dataset):
    """
    1. 错误检测评估：对测试集进行 tokenize 和标签对齐，
       利用 Trainer 进行预测，再计算 Accuracy、Precision、Recall、F1 及分类报告。
    """
    tokenized_datasets = test_dataset.map(tokenize_and_align_labels, batched=True)
    data_collator = DataCollatorForTokenClassification(detection_tokenizer)
    trainer = Trainer(model=detection_model, tokenizer=detection_tokenizer, data_collator=data_collator)
    pred = trainer.predict(tokenized_datasets)
    return compute_token_classification_metrics(pred.predictions, pred.label_ids)


def evaluate_correction_perfect(test_dataset, corrector):
    """
    2. 纠错评估（完美检测）：基于 ground truth 生成 masked_input，
       仅对含错误的样本进行评估，统计平均编辑距离和完全一致准确率。
    """
    total_samples = 0
    total_edit_distance = 0
    exact_match_count = 0
    for sample in test_dataset:
        if sample.get("error_count", 0) == 0:
            continue
        masked_input = create_soft_masked_data(sample)["masked_input"]
        corrected_text, _ = iterative_correction(
            corrector, masked_input, sample["output"],
            max_iters=10, alpha=0.7, beta=0.3, gamma=0.3
        )
        edit_distance = Levenshtein.distance(corrected_text, sample["output"])
        total_edit_distance += edit_distance
        total_samples += 1
        if corrected_text == sample["output"]:
            exact_match_count += 1
    if total_samples == 0:
        return None
    avg_edit_distance = total_edit_distance / total_samples
    exact_match_accuracy = exact_match_count / total_samples * 100
    return total_samples, exact_match_accuracy, avg_edit_distance


def evaluate_correction_actual(test_dataset, detection_model, detection_tokenizer, corrector):
    """
    3. 纠错评估（实际检测）：
       仅对含错误的样本进行评估。
    """
    total_samples = 0
    total_edit_distance = 0
    exact_match_count = 0
    for sample in test_dataset:
        if sample.get("error_count", 0) == 0:
            continue
        input_text = sample["input"]
        true_output = sample["output"]
        pred_labels = detect_errors_in_sentence(input_text, detection_model, detection_tokenizer)
        masked_input = create_masked_text_from_predictions(input_text, pred_labels)
        corrected_text, _ = iterative_correction(
            corrector, masked_input, true_output,
            max_iters=10, alpha=0.7, beta=0.3, gamma=0.3
        )
        edit_distance = Levenshtein.distance(corrected_text, true_output)
        total_edit_distance += edit_distance
        total_samples += 1
        if corrected_text == true_output:
            exact_match_count += 1

    if total_samples == 0:
        return None
    avg_edit_distance = total_edit_distance / total_samples
    exact_match_accuracy = exact_match_count / total_samples * 100
    return total_samples, exact_match_accuracy, avg_edit_distance


#############################################
# 4. 新增：对整份测试集(含错误/无错)的“实际检测+纠错”评估

def evaluate_correction_actual_full(test_dataset, detection_model, detection_tokenizer, corrector):
    """
    4. 纠错评估（实际检测-全量）：
       对整个测试集(含 error_count=0 和 error_count>0 的样本)执行：
         -> 错误检测 -> 生成 mask -> 迭代纠错 -> 与真实 output 比较
       返回(总样本数, 完全一致准确率, 平均字符级编辑距离)。
    """
    total_samples = len(test_dataset)
    total_edit_distance = 0
    exact_match_count = 0

    for sample in test_dataset:
        input_text = sample["input"]
        true_output = sample["output"]

        # 1. 检测
        pred_labels = detect_errors_in_sentence(input_text, detection_model, detection_tokenizer)
        # 2. 根据预测生成 mask 句
        masked_input = create_masked_text_from_predictions(input_text, pred_labels)
        # 3. 纠错
        corrected_text, _ = iterative_correction(
            corrector, masked_input, true_output,
            max_iters=10, alpha=0.7, beta=0.3, gamma=0.3
        )
        # 4. 统计
        edit_distance = Levenshtein.distance(corrected_text, true_output)
        total_edit_distance += edit_distance
        if corrected_text == true_output:
            exact_match_count += 1

    avg_edit_distance = total_edit_distance / total_samples
    exact_match_accuracy = exact_match_count / total_samples * 100
    return total_samples, exact_match_accuracy, avg_edit_distance


#############################################
# 5. 主流程：加载数据、模型，进行评估

def main():
    # 加载测试数据集（请确保路径正确）
    data_files = {"test": "../dataset/law_correction_test.jsonl"}
    dataset = load_dataset("json", data_files=data_files)
    test_dataset = dataset["test"]
    print(f"测试样本总数: {len(test_dataset)}")

    # ----------------------------
    # 加载错误检测模型和 tokenizer（基于 MacBERT）
    global detection_tokenizer  # 在 tokenize_and_align_labels 中使用
    error_model_name = "hfl/chinese-macbert-base"
    detection_model_path = "../macbert-error-detect/checkpoint-294"
    detection_tokenizer = AutoTokenizer.from_pretrained(error_model_name)
    try:
        detection_model = AutoModelForTokenClassification.from_pretrained(
            detection_model_path, num_labels=2)
    except Exception as e:
        print("未找到微调后的错误检测模型，使用预训练模型作为替代。")
        detection_model = AutoModelForTokenClassification.from_pretrained(
            error_model_name, num_labels=2)
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        detection_model.to("cuda")
        print("Device set to use cuda:0")
    else:
        print("Device set to CPU")
    detection_model.eval()

    # ----------------------------
    # 加载纠错模型和 tokenizer（Soft-Masked 模型），构造 fill-mask pipeline
    correction_model_path = "../macbert-soft-masked/checkpoint-294"
    correction_tokenizer = AutoTokenizer.from_pretrained(correction_model_path)
    corrector = pipeline("fill-mask", model=correction_model_path, tokenizer=correction_tokenizer)
    print("模型加载完毕。\n")

    # 1. 错误检测评估
    print("========== 错误检测评估 ==========")
    accuracy, precision, recall, f1, report = evaluate_detection(
        detection_model, detection_tokenizer, test_dataset)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Classification Report:")
    print(report)

    # 2. 纠错评估（完美检测） - 仅有错样本
    print("\n========== 纠错评估（完美检测） - 仅含错误样本 ==========")
    result_perfect = evaluate_correction_perfect(test_dataset, corrector)
    if result_perfect:
        samples_perf, exact_match_perf, avg_edit_perf = result_perfect
        print(f"评估样本数（含错误）： {samples_perf}")
        print(f"完全一致准确率: {exact_match_perf:.2f}%")
        print(f"平均字符级编辑距离: {avg_edit_perf:.2f}")
    else:
        print("无错误样本，无法评估纠错（完美检测）效果。")

    # 3. 纠错评估（实际检测） - 仅有错样本
    print("\n========== 纠错评估（实际检测） - 仅含错误样本 ==========")
    result_actual = evaluate_correction_actual(
        test_dataset, detection_model, detection_tokenizer, corrector)
    if result_actual:
        samples_act, exact_match_act, avg_edit_act = result_actual
        print(f"评估样本数（含错误）： {samples_act}")
        print(f"完全一致准确率: {exact_match_act:.2f}%")
        print(f"平均字符级编辑距离: {avg_edit_act:.2f}")
    else:
        print("无错误样本，无法评估纠错（实际检测）效果。")

    # 4. 纠错评估（实际检测-全量） - 面向所有500条
    print("\n========== 纠错评估（实际检测-全量） - 全部测试集 ==========")
    total_samples, exact_match_full, avg_edit_full = evaluate_correction_actual_full(
        test_dataset, detection_model, detection_tokenizer, corrector
    )
    print(f"评估样本总数： {total_samples}")
    print(f"整句完全一致准确率: {exact_match_full:.2f}%")
    print(f"平均字符级编辑距离: {avg_edit_full:.2f}")


if __name__ == "__main__":
    main()
