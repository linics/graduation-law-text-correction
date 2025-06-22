# total_test.py - 完整六组对照实验版本
# 包含：错误检测评估、错误纠正评估（仅错误样本）、完整流程评估（全样本）
# 模型对比：未微调 vs 微调；纠错方式对比：不纠错、零权重、模型得分、最优权重

import os
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    pipeline,
    DataCollatorForTokenClassification,
    Trainer,
    logging
)
import Levenshtein
import pypinyin
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from similarity_utils import get_all_candidate_scores


logging.set_verbosity_error()

# ========== 工具函数 ==========

def compute_char_accuracy(pred, truth):
    max_len = max(len(pred), len(truth))
    pred = pred.ljust(max_len)
    truth = truth.ljust(max_len)
    return sum(p == t for p, t in zip(pred, truth)) / max_len

def find_error_positions(input_text, output_text):
    min_length = min(len(input_text), len(output_text))
    error_positions = [1 if input_text[i] != output_text[i] else 0 for i in range(min_length)]
    if len(input_text) > min_length:
        error_positions.extend([0] * (len(input_text) - min_length))
    return error_positions

def tokenize_and_align_labels(examples, tokenizer):
    inputs_split = [list(s) for s in examples["input"]]
    tokenized_inputs = tokenizer(
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
        if examples.get("error_count", [0]*len(examples["input"]))[i] == 0:
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

def detect_errors(sentence, model, tokenizer):
    inputs = list(sentence)
    raw = tokenizer([inputs], is_split_into_words=True, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
    word_ids = raw.word_ids(batch_index=0)
    device = next(model.parameters()).device
    raw = {k: v.to(device) for k, v in raw.items()}
    outputs = model(**raw)
    preds = outputs.logits.detach().cpu().numpy()
    preds = np.argmax(preds, axis=2)[0]
    pred_labels = []
    prev = None
    for word_idx, pred in zip(word_ids, preds):
        if word_idx is None:
            continue
        elif word_idx != prev:
            pred_labels.append(pred)
        prev = word_idx
    return pred_labels

def create_masked_text(sentence, pred_labels):
    return ''.join('[MASK]' if l == 1 else c for c, l in zip(sentence, pred_labels))

def iterative_correction(corrector, masked_text, target_text, alpha=1.0, beta=0.0, gamma=0.0):
    result = masked_text
    iters = 0
    while '[MASK]' in result and iters < 10:
        try:
            preds = corrector(result)
            candidates = preds[0] if isinstance(preds[0], list) else preds
            idx = result.index('[MASK]')
            true_char = target_text[idx] if idx < len(target_text) else ''
            ranked = get_all_candidate_scores(candidates, true_char, alpha, beta, gamma)
            result = result.replace('[MASK]', ranked[0][0], 1)
        except Exception as e:
            break
        iters += 1
    return result

def evaluate_correction(test_set, detector, tokenizer, corrector, mode, alpha=1.0, beta=0.0, gamma=0.0):
    total = 0
    match = 0
    ed_sum = 0
    acc_sum = 0
    for sample in test_set:
        input_text = sample["input"]
        output_text = sample["output"]
        if mode == "no_correction":
            pred = input_text
        elif mode == "zero_shot":
            diff_positions = find_error_positions(input_text, output_text)
            masked = create_masked_text(input_text, diff_positions)
            pred = iterative_correction(corrector, masked, output_text, alpha, beta, gamma)
        else:
            pred_labels = detect_errors(input_text, detector, tokenizer)
            masked = create_masked_text(input_text, pred_labels)
            pred = iterative_correction(corrector, masked, output_text, alpha, beta, gamma)
        total += 1
        match += (pred == output_text)
        ed_sum += Levenshtein.distance(pred, output_text)
        acc_sum += compute_char_accuracy(pred, output_text)
    return total, match/total*100, ed_sum/total, acc_sum/total*100

def compute_detection_metrics(model, tokenizer, dataset):
    tokenized = dataset.map(lambda x: tokenize_and_align_labels(x, tokenizer), batched=True)
    data_collator = DataCollatorForTokenClassification(tokenizer)
    trainer = Trainer(model=model, tokenizer=tokenizer, data_collator=data_collator)
    output = trainer.predict(tokenized)
    acc, prec, rec, f1, report = compute_token_classification_metrics(output.predictions, output.label_ids)
    return acc, prec, rec, f1, report

def compute_token_classification_metrics(preds, labels):
    preds = np.argmax(preds, axis=2)
    true_preds, true_labels = [], []
    for pred_seq, label_seq in zip(preds, labels):
        for p, l in zip(pred_seq, label_seq):
            if l != -100:
                true_preds.append(p)
                true_labels.append(l)
    acc = accuracy_score(true_labels, true_preds)
    prec = precision_score(true_labels, true_preds, average='binary')
    rec = recall_score(true_labels, true_preds, average='binary')
    f1 = f1_score(true_labels, true_preds, average='binary')
    report = classification_report(true_labels, true_preds, target_names=["correct", "error"])
    return acc, prec, rec, f1, report

# ========== 主函数 ==========
def main():
    # 加载数据
    dataset = load_dataset("json", data_files={"test": "../dataset/law_correction_test.jsonl"})
    test = dataset["test"]
    test_errors = [s for s in test if s["error_count"] > 0]

    # 加载模型
    tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-macbert-base")
    detector_pre = AutoModelForTokenClassification.from_pretrained("hfl/chinese-macbert-base", num_labels=2)
    detector_ft = AutoModelForTokenClassification.from_pretrained("../macbert-error-detect/checkpoint-294", num_labels=2)
    corrector_pre = pipeline("fill-mask", model="hfl/chinese-macbert-base", tokenizer=tokenizer)
    corrector_ft = pipeline("fill-mask", model="../macbert-soft-masked/checkpoint-294", tokenizer=tokenizer)

    print("========== 错误检测评估 ==========")
    for name, model in zip(["预训练模型", "微调模型"], [detector_pre, detector_ft]):
        acc, prec, rec, f1, report = compute_detection_metrics(model, tokenizer, test)
        print(f"[{name}] 准确率: {acc:.4f} 精确率: {prec:.4f} 召回率: {rec:.4f} F1: {f1:.4f}")

    print("\n========== 错误纠正评估（仅错误样本） ==========")
    configs = [
        ("不进行纠错（Baseline）", "no_correction", corrector_ft, 0, 0, 0),
        ("未微调纠错（零权重）", "zero_shot", corrector_pre, 0, 0, 0),
        ("微调纠错（模型得分）", "correct", corrector_ft, 1.0, 0.0, 0.0),
        ("微调纠错（最优权重）", "correct", corrector_ft, 0.4, 0.4, 0.1),
    ]
    for name, mode, corr, a, b, g in configs:
        total, em, ed, acc = evaluate_correction(test_errors, detector_ft, tokenizer, corr, mode, a, b, g)
        print(f"{name:<30s} 样本数: {total:<4d} EM: {em:.2f}% ED: {ed:.2f} 字符Acc: {acc:.2f}%")

    print("\n========== 完整流程评估（全样本） ==========")
    full_configs = [
        ("未微调检测+未微调纠错（零权重）", detector_pre, corrector_pre, 0, 0, 0),
        ("微调检测+未微调纠错（零权重）", detector_ft, corrector_pre, 0, 0, 0),
        ("微调检测+微调纠错（最优权重）", detector_ft, corrector_ft, 0.4, 0.4, 0.1),
    ]
    for name, det, corr, a, b, g in full_configs:
        total, em, ed, acc = evaluate_correction(test, det, tokenizer, corr, "correct", a, b, g)
        print(f"{name:<30s} 样本数: {total:<4d} EM: {em:.2f}% ED: {ed:.2f} 字符Acc: {acc:.2f}%")

if __name__ == "__main__":
    main()
