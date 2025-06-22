# 中文法律文本错误检测与纠错系统

## 一、项目简介
本项目针对中文法律文本中的拼写错误、形近字和同音字问题，基于 MacBERT 构建了一个端到端的纠错系统。系统由错误检测模型和软掩码纠错模型组成，并融合语言模型置信度、拼音相似度和字形相似度三种评分因子，实现高精度的候选字符排序。前端采用 Streamlit 提供交互式界面，支持单句输入与批量处理。

## 二、目录结构
```
the_graduation_design/
│
├─ dataset/                      # EC-LAW 数据集（train/valid/test splits）
│
├─ macbert-error-detect/         # 错误检测模型微调输出
│
├─ macbert-soft-masked/          # 软掩码纠错模型微调输出
│
├─ train_test_scripts/
│   └─ tmp_trainer/
│       ├─ compare_with_pycorrector.py    # 与 pycorrector 对比脚本
│       ├─ macbert_detect_train.py        # 错误检测模型训练脚本
│       ├─ param_search.py                # 参数网格搜索脚本
│       ├─ softmasked_correct_train.py    # 软掩码纠错模型训练脚本
│       └─ total_test.py                  # 六组对照实验评估脚本
│
├─ .gitignore
├─ app.py                    # Streamlit 前端应用
├─ README.md
└─ requirements.txt
```

## 三、环境依赖
```bash
python3 -m venv venv
source venv/bin/activate     # Linux/Mac
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

主要依赖包括：`transformers`、`datasets`、`torch`、`streamlit`、`pypinyin`、`cnradical`、`python-Levenshtein`、`scikit-learn`、`pandas`。

## 四、数据集准备
请将 ECSpell 中的法律领域子集（EC-LAW）文件划分数据集并放入 `dataset/`，包括：
- `law_correction_train_split.jsonl`
- `law_correction_valid.jsonl`
- `law_correction_test.jsonl`

## 五、模型训练

### 1. 错误检测模型微调
```bash
python train_test_scripts/tmp_trainer/macbert_detect_train.py
```
输出模型至 `macbert-error-detect/`。

### 2. 纠错模型微调（Soft-Masked MLM）
```bash
python train_test_scripts/tmp_trainer/softmasked_correct_train.py
```
输出模型至 `macbert-soft-masked/`。

## 六、参数网格搜索
```bash
python train_test_scripts/tmp_trainer/param_search.py
```
在验证集上搜索最优 `(α, β, γ)`，结果保存为 `parameter_search_results.csv`。

## 七、对照实验评估

### 1. 错误检测评估（2 组）
- 预训练模型
- 微调模型

### 2. 错误纠正评估（仅错误样本，4 组）
- 不进行纠错（Baseline）
- 未微调纠错（零权重）
- 微调纠错（模型得分）
- 微调纠错（融合权重）

### 3. 全流程评估（全样本，3 组）
- 未微调检测 + 未微调纠错（零权重）
- 微调检测 + 未微调纠错（零权重）
- 微调检测 + 微调纠错（融合权重）

### 4. 与 pycorrector-MacBERT 对比（2 组）
在相同测试集上，将 **pycorrector-MacBERT** 模型与本系统最优配置（微调检测 + 微调纠错，融合权重 α=0.4, β=0.4, γ=0.1）进行对比。实验结果如下：

| 系统                                 | 完全匹配率 (%) | 平均编辑距离 | 字符准确率 (%) |
|--------------------------------------|---------------:|-------------:|--------------:|
| pycorrector-MacBERT                  |          60.80 |         0.60 |         97.77 |
| 本系统（微调检测+微调纠错，融合权重）|          79.60 |         0.29 |         98.61 |

以上结果验证了融合打分策略与领域微调的有效性。

## 八、启动前端应用
```bash
python -m streamlit run app.py
```

## 九、核心接口示例
```python
from text_correction_tool import TextCorrectionTool

tool = TextCorrectionTool(
    detection_model_id="linics/detection-model",
    correction_model_id="linics/correction-model",
    device="gpu"
)
res = tool.correct_text("输入含错法律文本", alpha=0.4, beta=0.4, gamma=0.1)
print(res)
```

## 十、说明
- 本 README 旨在说明项目结构、使用方式及实验对比。
- 本项目已同步至 GitHub 仓库: https://github.com/linics/graduation-law-text-correction

## 十一、运行与数据库初始化
```bash
pip install -r requirements.txt
python migrations/001_initial.py  # 初始化数据库
streamlit run app.py
```

若出现依赖安装失败，请确认已离线下载所需 whl 包，并通过 `pip install <file>` 方式安装。

## 常见问题
- **模型加载慢**：首次运行需从本地或预训练权重加载，可预先下载至本地。
- **无法写入数据库**：检查 `app.db` 是否有写权限。

