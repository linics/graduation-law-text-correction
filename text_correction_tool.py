# text_correction_tool.py

import os
import torch
import numpy as np
import pypinyin
import Levenshtein
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from similarity_utils import get_all_candidate_scores
from term_filter import load_terms, filter_candidates


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
        # 术语词表
        self.terms = load_terms()

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

    def iterative_correction(self, masked_text, target_text, max_iters=10, alpha=None, beta=None, gamma=None, debug=False, legal_only=False):
        """
        迭代纠错：对 ``masked_text`` 不断填充 ``[MASK]``，直到文本中不再包含 ``[MASK]``
        或达到 ``max_iters`` 次。

        每次生成候选词后，会通过 ``term_filter.filter_candidates`` 校验其是否合法
        术语，只有被标记为合法的候选才参与后续打分与替换。

        ``legal_only=True`` 时将强制仅在术语词表命中的候选中选择，若无命中则回填
        原字符，不进入 General Branch。

        如果 ``debug=True``，则返回候选词得分列表，前端可用于展示；同时暴露替换
        时匹配到的术语及分支信息，便于高亮显示。

        返回值：
          - debug=False  -> ``(corrected_sentence, matched_terms, branch_history)``
          - debug=True   -> ``(corrected_sentence, predictions_list, matched_terms, branch_history)``
        """
        alpha = self.default_alpha if alpha is None else alpha
        beta = self.default_beta if beta is None else beta
        gamma = self.default_gamma if gamma is None else gamma

        corrected_sentence = masked_text
        predictions_list = []  # 用于保存每次迭代的候选词及分数信息
        matched_terms = []      # 每次替换命中的术语
        branch_history = []     # 记录每次迭代走的分支
        iters = 0
        while "[MASK]" in corrected_sentence and iters < max_iters:
            try:
                results = self.corrector(corrected_sentence)
                candidate_list = results[0] if isinstance(results[0], list) else results
                mask_index = corrected_sentence.index("[MASK]")
                target_char = target_text[mask_index] if mask_index < len(target_text) else ""

                # 候选字符串列表
                cand_tokens = [c["token_str"] for c in candidate_list]
                term_info = filter_candidates(cand_tokens, self.terms)
                valid_candidates = []
                for cand, info in zip(candidate_list, term_info):
                    cand = {**cand, "matched_term": info["matched_term"], "flag": info["flag"]}
                    if info["flag"]:
                        valid_candidates.append(cand)

                # 记录分支
                branch = "Precision" if valid_candidates else "General"
                if legal_only:
                    branch = "Precision"
                branch_history.append(branch)

                # 强制精确分支且没有合法候选，则直接回填原字符
                if legal_only and not valid_candidates:
                    if debug:
                        predictions_list.append([])
                    matched_terms.append(None)
                    corrected_sentence = corrected_sentence.replace("[MASK]", target_char, 1)
                    iters += 1
                    continue

                # 若没有合法候选，则退回原始候选列表
                source_for_score = valid_candidates if valid_candidates else candidate_list
                candidates_with_scores = []
                raw_scores = get_all_candidate_scores(source_for_score, target_char,
                                                      alpha=alpha, beta=beta, gamma=gamma)
                for score_item, cand in zip(raw_scores, source_for_score):
                    token, combined, model_score, pinyin_sim, shape_sim = score_item
                    candidates_with_scores.append(
                        (token, combined, model_score, pinyin_sim, shape_sim, cand.get("matched_term"))
                    )

                if debug:
                    predictions_list.append(candidates_with_scores)

                top_candidate, _, _, _, _, top_match = candidates_with_scores[0]
                matched_terms.append(top_match)
                corrected_sentence = corrected_sentence.replace("[MASK]", top_candidate, 1)
            except Exception as e:
                print("fill-mask 出错:", e)
                break
            iters += 1
        if debug:
            return corrected_sentence, predictions_list, matched_terms, branch_history
        else:
            return corrected_sentence, matched_terms, branch_history

    ################################################################################
    # 4. 对外暴露的核心接口
    ################################################################################
    def correct_text(self, sentence, alpha=None, beta=None, gamma=None, legal_only=False):
        """
        综合接口：先对输入句子进行错误检测 → 生成 masked 文本 → 迭代纠错，
        返回最终纠错后的句子。
        """
        alpha = self.default_alpha if alpha is None else alpha
        beta = self.default_beta if beta is None else beta
        gamma = self.default_gamma if gamma is None else gamma

        pred_labels = self.detect_errors_in_sentence(sentence)
        masked_text = self.create_masked_text_from_predictions(sentence, pred_labels)
        corrected_sentence, _, _ = self.iterative_correction(
            masked_text, sentence,
            alpha=alpha, beta=beta, gamma=gamma,
            debug=False,
            legal_only=legal_only
        )
        return corrected_sentence
