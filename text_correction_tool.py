# text_correction_tool.py

import os
import torch
import numpy as np
import pypinyin
import Levenshtein
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from similarity_utils import get_all_candidate_scores


################################################################################
# 1. 全局加载或初始化
################################################################################

class TextCorrectionTool:
    """
    用于加载检测模型 & 纠错模型，并提供检测 + 纠错的接口。
    """

    def __init__(self,
                 detection_model_id="linics/detection-model",  # 你的检测模型ID
                 correction_model_id="linics/correction-model",  # 你的纠错模型ID
                 device="cuda"):
        # 加载错误检测模型
        self.detection_tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-macbert-base")
        self.detection_model = AutoModelForTokenClassification.from_pretrained(
            detection_model_id,
            num_labels=2
        )
        if device == "cuda" and torch.cuda.is_available():
            self.detection_model.to("cuda")
        else:
            self.detection_model.to("cpu")

        # 加载纠错模型
        self.correction_tokenizer = AutoTokenizer.from_pretrained(correction_model_id)
        self.corrector = pipeline("fill-mask", model=correction_model_id, tokenizer=self.correction_tokenizer,
                                  device=0 if device == "cuda" else -1)
        self.default_alpha = 0.4
        self.default_beta = 0.4
        self.default_gamma = 0.2

    ################################################################################
    # 2. 检测相关
    ################################################################################
    def detect_errors_in_sentence(self, sentence, max_length=128):
        """
        对单个句子进行错误检测，返回每个字符的预测标签列表（0 表示正确，1 表示错误）。
        将输入句子按字符拆分，并利用检测模型预测每个字符的标签。
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
        # 获取 tokenized 结果中每个 token 对应的 word_ids
        word_ids = raw_tokenized.word_ids(batch_index=0)

        # 将张量转移到模型所在设备
        device = next(self.detection_model.parameters()).device
        tokenized_input = {k: v.to(device) for k, v in raw_tokenized.items()}

        with torch.no_grad():
            outputs = self.detection_model(**tokenized_input)
        predictions = outputs.logits.detach().cpu().numpy()
        # 对每个 token 选择得分最大的类别
        predictions = np.argmax(predictions, axis=2)[0]

        # 对齐 word_ids，只保留每个单词第一个 token 的预测结果
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
        根据预测标签，将预测为错误的位置替换为 [MASK]，否则保留原字符。
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

    def iterative_correction(self, masked_text, target_text, max_iters=10, alpha=None, beta=None, gamma=None, debug=False):
        """
        迭代纠错：对 masked_text 不断填充 [MASK]，直到文本中不再包含 [MASK] 或达到最大迭代次数。
        如果 debug=True，则同时返回每一步的候选词列表。
        返回：
          - 如果 debug 为 False，返回最终纠错后的文本；
          - 如果 debug 为 True，返回 (corrected_sentence, predictions_list)。
        """
        alpha = self.default_alpha if alpha is None else alpha
        beta = self.default_beta if beta is None else beta
        gamma = self.default_gamma if gamma is None else gamma

        corrected_sentence = masked_text
        predictions_list = []  # 用于保存每次迭代的候选词及分数信息
        iters = 0
        while "[MASK]" in corrected_sentence and iters < max_iters:
            try:
                results = self.corrector(corrected_sentence)
                candidate_list = results[0] if isinstance(results[0], list) else results
                mask_index = corrected_sentence.index("[MASK]")
                target_char = target_text[mask_index] if mask_index < len(target_text) else ""
                candidates_with_scores = get_all_candidate_scores(candidate_list, target_char,
                                                                  alpha=alpha, beta=beta, gamma=gamma)
                if debug:
                    predictions_list.append(candidates_with_scores)
                top_candidate = candidates_with_scores[0][0]
                corrected_sentence = corrected_sentence.replace("[MASK]", top_candidate, 1)
            except Exception as e:
                print("fill-mask 出错:", e)
                break
            iters += 1
        if debug:
            return corrected_sentence, predictions_list
        else:
            return corrected_sentence

    ################################################################################
    # 4. 对外暴露的核心接口
    ################################################################################
    def correct_text(self, sentence, alpha=None, beta=None, gamma=None):
        """
        综合接口：先对输入句子进行错误检测 → 生成 masked 文本 → 迭代纠错，
        返回最终纠错后的句子。
        """
        alpha = self.default_alpha if alpha is None else alpha
        beta = self.default_beta if beta is None else beta
        gamma = self.default_gamma if gamma is None else gamma

        pred_labels = self.detect_errors_in_sentence(sentence)
        masked_text = self.create_masked_text_from_predictions(sentence, pred_labels)
        corrected_sentence = self.iterative_correction(masked_text, sentence,
                                                       alpha=alpha, beta=beta, gamma=gamma)
        return corrected_sentence
