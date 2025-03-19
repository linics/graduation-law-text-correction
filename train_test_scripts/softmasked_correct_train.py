import re
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM, Trainer, TrainingArguments, DataCollatorWithPadding
import torch


# ----------------------------
# 1. 数据生成：构造 Soft-Masked 数据
def create_soft_masked_data(example):
    """
    针对每个样本，逐字符对比 input 和 output：
      - 如果字符不一致，则输出 “[MASK]”
      - 否则输出该字符
    生成字段 "masked_input"。
    """
    input_text = example["input"]
    output_text = example["output"]
    # 如果没有错误，则直接返回原句
    if example["error_count"] == 0:
        return {"masked_input": input_text}

    tokens = []
    for a, b in zip(input_text, output_text):
        if a != b:
            tokens.append("[MASK]")
        else:
            tokens.append(a)
    # 如果 input 比 output 长（或反之），这里暂简单按 zip 结果处理
    masked_input = "".join(tokens)
    return {"masked_input": masked_input}


# ----------------------------
# 2. 加载原始数据并生成 Soft-Masked 数据
data_files = {
    "train": "dataset/law_correction_train_split.jsonl",
    "validation": "dataset/law_correction_valid.jsonl",
    "test": "dataset/law_correction_test.jsonl"
}
raw_dataset = load_dataset("json", data_files=data_files)
print("原始数据集 keys:", raw_dataset.keys())

# 生成 soft-masked 数据（只为 error_count>0 的样本会改变，否则 masked_input 与 input 相同）
masked_dataset = raw_dataset.map(create_soft_masked_data)
print("示例 soft-masked 数据:", masked_dataset["train"][0])

# ----------------------------
# 3. Tokenization 及对齐：逐字符处理，保证 “[MASK]” 不被拆分
model_name = "hfl/chinese-macbert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)


def tokenize_for_softmasked(example):
    """
    对于每个样本：
      - 构造 token 列表：遍历 input 与 output，若字符不一致，输出 “[MASK]”，否则输出原字符。
      - 使用 tokenizer(token_list, is_split_into_words=True) 得到 input_ids 等。
      - 对齐标签：如果对应 token 为 “[MASK]”，则 label 为 output 中对应字符转换的 token id；否则为 -100。
    """
    input_text = example["input"]
    output_text = example["output"]
    # 构造 token 列表：如果 error_count==0，token_list = list(input_text)
    # 如果 error_count>0，替换不同的字符为 "[MASK]"
    if example["error_count"] == 0:
        token_list = list(input_text)
    else:
        token_list = []
        for a, b in zip(input_text, output_text):
            if a != b:
                token_list.append("[MASK]")
            else:
                token_list.append(a)
    # tokenize
    tokenized = tokenizer(token_list, is_split_into_words=True, truncation=True, max_length=128, padding="max_length")

    # 对齐标签
    labels = []
    # 得到每个 token 对应原始 token_list 的索引
    word_ids = tokenized.word_ids()
    for idx in word_ids:
        if idx is None:
            labels.append(-100)
        else:
            # 如果 token_list[idx] 是 “[MASK]”，则 label 应该为 output_text[idx] 转换的 token id
            if token_list[idx] == "[MASK]":
                # 对 output_text[idx] 单独 tokenize，应当返回单个 token
                out_token = tokenizer.tokenize(output_text[idx])
                if out_token:
                    label_id = tokenizer.convert_tokens_to_ids(out_token)[0]
                else:
                    label_id = -100
                labels.append(label_id)
            else:
                labels.append(-100)
    tokenized["labels"] = labels
    return tokenized


# 对 train、validation、test 分别处理
tokenized_datasets = masked_dataset.map(tokenize_for_softmasked, batched=False)
print("预处理后示例：", tokenized_datasets["train"][0])

# 使用 DataCollatorWithPadding 来动态 padding
data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

# ----------------------------
# 4. 训练 Soft-Masked MacBERT 模型
model = AutoModelForMaskedLM.from_pretrained(model_name)

training_args = TrainingArguments(
    output_dir="./macbert-soft-masked",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=50,
    save_strategy="epoch",
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()
eval_results = trainer.evaluate(tokenized_datasets["validation"])
print("评估结果：", eval_results)

# ----------------------------
# 5. 测试 Soft-Masked 纠错效果
from transformers import pipeline

corrector = pipeline("fill-mask", model=model, tokenizer=tokenizer)

# 从测试集中选择一个 error_count > 0 的样本
sample = None
for ex in masked_dataset["test"]:
    if ex["error_count"] > 0:
        sample = ex
        break

if sample is None:
    print("测试集中无错误样本，无法测试纠错效果。")
else:
    print("原始 input:", sample["input"])
    print("正确 output:", sample["output"])
    print("生成 masked_input:", sample["masked_input"])
    # 使用 corrector 对 masked_input 进行预测
    results = corrector(sample["masked_input"])
    print("\n🔍 纠错候选结果：")
    for res in results:
        print(f"候选替换: {res['token_str']} (置信度: {res['score']:.4f})")
