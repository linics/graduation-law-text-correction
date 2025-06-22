# 中文法律文本纠错系统

## 项目概览
本项目基于 MacBERT 模型构建，针对法律场景的中文文本进行错误检测与纠正。仓库内提供了模型训练脚本、参数搜索工具以及基于 Streamlit 的演示界面。

## 目录说明
- `dataset/`：训练、验证与测试数据
- `services/`：数据库与认证相关代码
- `pages/`：Streamlit 页面脚本
- `train_test_scripts/`：训练与评估脚本
- 其余文件：核心逻辑实现与启动脚本

## 快速开始
1. 准备 Python 3.9 以上环境，并安装依赖：
   ```bash
   pip install -r requirements.txt  # 请提前准备离线包
   ```
2. 初始化数据库：
   ```bash
   python migrations/001_initial.py
   ```
3. 启动演示界面：
   ```bash
   streamlit run 首页.py
   ```

## 模型训练
- 错误检测模型：
  ```bash
  python train_test_scripts/macbert_detect_train.py
  ```
- 纠错模型：
  ```bash
  python train_test_scripts/softmasked_correct_train.py
  ```
- 参数搜索：
  ```bash
  python train_test_scripts/param_search.py
  ```

## 示例调用
```python
from text_correction_tool import TextCorrectionTool

tool = TextCorrectionTool(device="cuda")
result = tool.correct_text("示例文本")
print(result)
```

## 说明
本仓库同步自 [GitHub](https://github.com/linics/graduation-law-text-correction)。更多实验细节可在 `train_test_scripts` 中找到。
