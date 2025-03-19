from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments, \
    DataCollatorForTokenClassification
from datasets import load_dataset

# 1. **加载数据**
data_files = {
    "train": "dataset/law_correction_train_split.jsonl",
    "validation": "dataset/law_correction_valid.jsonl",
    "test": "dataset/law_correction_test.jsonl"
}
dataset = load_dataset("json", data_files=data_files)

# 2. **加载 Tokenizer 和 Model**
model_name = "hfl/chinese-macbert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=2)  # 2类：正确(0) / 错误(1)


# 3. **错误位置标注**
def find_error_positions(input_text, output_text):
    """
    识别 input 和 output 之间的不同字符位置，返回错误位置列表
    """
    min_length = min(len(input_text), len(output_text))
    error_positions = [1 if i < min_length and input_text[i] != output_text[i] else 0 for i in range(len(input_text))]

    # 如果 output 比 input 长，可能是少字错误
    if len(input_text) < len(output_text):
        error_positions.extend([1] * (len(output_text) - len(input_text)))

    return error_positions


# 4. **数据预处理**
def tokenize_and_align_labels(examples):
    """
    处理句子级数据：
    - Tokenize 句子
    - 生成与 Tokenizer 对齐的 Label
    """
    tokenized_inputs = tokenizer(examples["input"], padding="max_length", truncation=True, max_length=128)

    labels = []
    for i in range(len(examples["input"])):
        input_text = examples["input"][i]
        output_text = examples["output"][i]

        if examples["error_count"][i] == 0:
            # 无错误，全部标 0
            label_ids = [0] * len(input_text)
        else:
            # 计算错误位置
            label_ids = find_error_positions(input_text, output_text)

        # 确保与 tokenizer 结果对齐
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        aligned_labels = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                aligned_labels.append(-100)  # 填充部分不计算 loss
            elif word_idx != previous_word_idx:
                aligned_labels.append(label_ids[word_idx])  # 仅对第一个子词标注
            else:
                aligned_labels.append(-100)
            previous_word_idx = word_idx

        labels.append(aligned_labels)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


# 5. **处理数据**
tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)
data_collator = DataCollatorForTokenClassification(tokenizer)

# 6. **训练参数**
training_args = TrainingArguments(
    output_dir="./macbert-error-detect",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=50,
    save_strategy="epoch",
    save_total_limit=10,
)

# 7. **训练**
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()

# 8. **测试**
print("\n📊 开始测试模型...")
results = trainer.evaluate(tokenized_datasets["test"])
print(f"测试集结果: {results}")
