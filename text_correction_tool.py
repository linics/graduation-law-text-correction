# text_correction_tool.py

import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import Levenshtein
import pypinyin

################################################################################
# 1. 全局加载或初始化
################################################################################

class TextCorrectionTool:
    """
    用于加载检测模型 & 纠错模型，并提供检测 + 纠错的接口。
    """

    def __init__(self,
                 detection_model_path="./macbert-error-detect/checkpoint-294",
                 correction_model_path="./macbert-soft-masked/checkpoint-294",
                 device="cuda"  # or "cpu"
                 ):
        """
        初始化：加载检测模型、纠错模型
        """
        # 1) 加载错误检测模型
        self.detection_tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-macbert-base")
        self.detection_model = AutoModelForTokenClassification.from_pretrained(
            detection_model_path, num_labels=2
        )
        if device == "cuda" and torch.cuda.is_available():
            self.detection_model.to("cuda")
        self.detection_model.eval()

        # 2) 加载纠错模型 (soft-masked)
        self.correction_tokenizer = AutoTokenizer.from_pretrained(correction_model_path)
        self.corrector = pipeline("fill-mask", model=correction_model_path, tokenizer=self.correction_tokenizer,
                                  device=0 if device=="cuda" and torch.cuda.is_available() else -1)

    ################################################################################
    # 2. 检测相关
    ################################################################################
    def detect_errors_in_sentence(self, sentence, max_length=128):
        """
        对单个句子进行错误检测，返回每个字符的预测标签列表（0表示正确，1表示错误）。
        """
        inputs = list(sentence)
        raw_tokenized = self.detection_tokenizer(
            [inputs],
            is_split_into_words=True,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        word_ids = raw_tokenized.word_ids(batch_index=0)  # 获取 word_ids

        device = next(self.detection_model.parameters()).device
        tokenized_input = {k: v.to(device) for k, v in raw_tokenized.items()}

        with torch.no_grad():
            outputs = self.detection_model(**tokenized_input)
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

    def create_masked_text_from_predictions(self, sentence, pred_labels):
        """
        根据预测标签，将预测为错误的位置替换为 [MASK]。
        """
        masked_chars = []
        for char, label in zip(sentence, pred_labels):
            if label == 1:
                masked_chars.append("[MASK]")
            else:
                masked_chars.append(char)
        return "".join(masked_chars)

    ################################################################################
    # 3. 纠错相关
    ################################################################################
    def get_pinyin(self, text):
        """将文本转换为拼音字符串"""
        return "".join(pypinyin.lazy_pinyin(text))

    def compute_pinyin_similarity(self, candidate, target):
        """
        计算候选词与目标词的拼音相似度。
        """
        cand_pinyin = self.get_pinyin(candidate)
        target_pinyin = self.get_pinyin(target)
        if cand_pinyin == target_pinyin:
            return 1.0
        max_len = max(len(cand_pinyin), len(target_pinyin))
        sim = 1 - Levenshtein.distance(cand_pinyin, target_pinyin) / max_len
        return sim

    def compute_shape_similarity(self, candidate, target):
        """简单示例：若候选词与目标词完全一致返回1.0，否则0.5。"""
        return 1.0 if candidate == target else 0.5

    def get_all_candidate_scores(self, candidate_list, target, alpha=0.7, beta=0.3, gamma=0.3):
        """
        对候选列表计算综合得分：
          综合得分 = alpha * 模型得分 + beta * 拼音相似度 + gamma * 字形相似度
        """
        results = []
        for cand in candidate_list:
            model_score = cand["score"]
            pinyin_sim = self.compute_pinyin_similarity(cand["token_str"], target)
            shape_sim = self.compute_shape_similarity(cand["token_str"], target)
            combined = alpha * model_score + beta * pinyin_sim + gamma * shape_sim
            results.append((cand["token_str"], combined, model_score, pinyin_sim, shape_sim))
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def iterative_correction(self, masked_text, target_text, max_iters=10, alpha=0.7, beta=0.3, gamma=0.3):
        """
        迭代纠错：对 masked_text 不断填充 [MASK]，直到无 [MASK] 或达到 max_iters。
        """
        corrected_sentence = masked_text
        iters = 0
        while "[MASK]" in corrected_sentence and iters < max_iters:
            try:
                results = self.corrector(corrected_sentence)
                candidate_list = results[0] if isinstance(results[0], list) else results
                mask_index = corrected_sentence.index("[MASK]")
                target_char = target_text[mask_index] if mask_index < len(target_text) else ""
                candidates_with_scores = self.get_all_candidate_scores(candidate_list, target_char,
                                                                       alpha=alpha, beta=beta, gamma=gamma)
                top_candidate = candidates_with_scores[0][0]
                corrected_sentence = corrected_sentence.replace("[MASK]", top_candidate, 1)
            except Exception as e:
                print("fill-mask 出错:", e)
                break
            iters += 1
        return corrected_sentence

    ################################################################################
    # 4. 对外暴露的核心接口
    ################################################################################
    def correct_text(self, sentence, alpha=0.7, beta=0.3, gamma=0.3):
        """
        综合接口：先检测错误位置 → 生成 masked_text → 迭代纠错 → 返回纠错后文本
        """
        # 1. 检测
        pred_labels = self.detect_errors_in_sentence(sentence)
        # 2. 生成 mask
        masked_text = self.create_masked_text_from_predictions(sentence, pred_labels)
        # 3. 纠错
        corrected_sentence = self.iterative_correction(masked_text, sentence,
                                                       alpha=alpha, beta=beta, gamma=gamma)
        return corrected_sentence
