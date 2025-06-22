# 中文法律文本纠错系统

## 项目背景
本仓库提供一套面向法律文本的中文纠错方案，核心算法基于 MacBERT 模型。系统集成了
错误检测、MASK 生成及迭代纠正流程，并配套 Streamlit 前端界面以及用户配额管理。

## 代码结构
- `app.py`：应用入口，负责启动 Streamlit 并处理登录。
- `text_correction_tool.py`：封装检测、MASK 构造与纠错流程，提供 `correct_text` 等接口。
- `services/`：数据库相关工具，`auth.py` 内含 `register`、`login` 等用户管理函数。
- `pages/`：Streamlit 多页面脚本，包含历史记录与日志查看功能。
- `train_test_scripts/`：模型训练和参数搜索脚本。
- `dataset/`：示例数据及术语词表。

过去版本中存在 `auth_utils.py` 与 `the_graduation_design/` 目录，目前均已移除，
统一使用 `services/auth.py` 和仓库根目录下的模块即可。

## 环境准备
1. 安装 Python 3.9 及以上版本。
2. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```
3. 初始化数据库：
   ```bash
   python migrations/001_initial.py
   ```

## 启动示例
启动可视化界面：
```bash
streamlit run app.py
```
首次运行需在页面中注册账号，系统会在 SQLite 数据库中记录用户信息及操作日志。

## 核心接口示例
```python
from text_correction_tool import TextCorrectionTool

# 可指定检测模型与纠错模型名称
tool = TextCorrectionTool(device="cuda")
text = "我国实行劳动者每日工作时间不超过8小时"
result = tool.correct_text(text, alpha=0.4, beta=0.4, gamma=0.2)
print(result)
```

`TextCorrectionTool` 内部流程如下：
1. 调用检测模型标记疑似错字位置。
2. 根据标签生成含 `[MASK]` 的文本。
3. 迭代调用纠错模型，对每个 `[MASK]` 进行替换，直至收敛或达到最大轮数。

## 模型训练
- 运行 `train_test_scripts/macbert_detect_train.py` 训练检测模型。
- 运行 `train_test_scripts/softmasked_correct_train.py` 训练纠错模型。
- `train_test_scripts/param_search.py` 用于搜索最佳权重组合。

## 测试
本仓库提供基础单元测试，修改代码后请执行：
```bash
pytest -q
```
测试脚本位于 `tests/` 目录，已通过 `pytest.ini` 限制仅收集该目录中的文件。

## 许可证
本项目使用 MIT License，详见仓库根目录 `LICENSE` 文件。
