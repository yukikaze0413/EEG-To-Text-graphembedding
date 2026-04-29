# 基于图嵌入的 EEG-To-Text

本仓库实现了一个面向 ZuCo 数据集的图结构 EEG 到文本解码流程。当前模型 `GAET` 使用时序编码器和图 Transformer 编码句子级 EEG 特征，通过对比学习对齐 EEG 表征和文本表征，并使用学习得到的图 token 条件化冻结的 BART 解码器。

评估阶段使用 `model.generate(...)` 进行自回归生成，不使用 teacher forcing 下的 logits 直接解码。

## 项目结构

```text
.
├── config.py                         # 训练和评估命令行参数
├── data.py                           # ZuCo 数据集加载与 EEG 图构建
├── model_decoding.py                 # GAET 模型定义
├── train_decoding.py                 # 两阶段训练入口
├── eval_decoding.py                  # 文本生成与指标评估入口
├── environment.yml                   # Conda 环境配置
├── scripts/
│   ├── prepare_dataset.sh            # Linux/macOS 数据准备脚本
│   ├── prepare_dataset_windows.ps1   # Windows 数据准备脚本
│   ├── train_gaet.sh                 # 训练辅助脚本
│   └── eval_gaet.sh                  # 评估辅助脚本
└── util/
    ├── construct_dataset_mat_to_pickle_v1.py
    ├── construct_dataset_mat_to_pickle_v2.py
    └── data_loading_helpers_modified.py
```

## 环境配置

使用 `environment.yml` 创建 Conda 环境：

```bash
conda env create -f environment.yml
conda activate EEGToText
```

环境主要包含 PyTorch 1.9.0、CUDA 11.1、Transformers 4.6.1，以及 `evaluate`、`sacrebleu`、`rouge`、`jiwer` 等评估指标依赖。

如果在服务器上训练，请确认服务器的 GPU 驱动、CUDA 版本和 PyTorch 构建版本兼容。

## 数据准备

将 ZuCo 的 `.mat` 文件放到以下目录结构中：

```text
dataset/ZuCo/
├── task1-SR/Matlab_files/
├── task2-NR/Matlab_files/
├── task3-TSR/Matlab_files/
└── task2-NR-2.0/Matlab_files/
```

将 MATLAB 文件转换为 pickle 文件：

```bash
bash scripts/prepare_dataset.sh
```

Windows PowerShell 下运行：

```powershell
.\scripts\prepare_dataset_windows.ps1
```

生成的 pickle 文件会保存到：

```text
dataset/ZuCo/<task-name>/pickle/
```

## 训练

使用训练脚本：

```bash
bash scripts/train_gaet.sh task1_task2_task3 EEG 0
```

参数说明：

- `task1_task2_task3`：任务组合，例如 `task1_task2_task3` 或 `task1_task2_taskNRv2`。
- `EEG`：训练输入类型。`EEG` 表示真实 EEG，`noise` 表示噪声基线。
- `0`：可见 GPU 编号。

训练分为两个阶段：

1. 对齐阶段：训练时序编码器、图 Transformer 和对比学习头。
2. 生成阶段：冻结 BART，仅训练 graph-to-BART projector。

模型 checkpoint 会保存到：

```text
checkpoints/decoding/best/
checkpoints/decoding/last/
```

训练配置会保存到：

```text
config/decoding/
```

## 评估

使用评估脚本：

```bash
bash scripts/eval_gaet.sh task1_task2_task3 EEG EEG 0
```

参数说明：

- `task1_task2_task3`：任务组合。
- `EEG`：训练时使用的输入类型。
- `EEG`：测试时使用的输入类型。
- `0`：可见 GPU 编号。

生成文本会保存到：

```text
results/
```

指标结果会追加写入：

```text
score_results/
```

评估脚本会计算 BLEU、SacreBLEU、WER、CER 和 ROUGE。

## 直接运行命令

直接训练：

```bash
python train_decoding.py \
  --task_name task1_task2_task3 \
  --train_input EEG \
  --data_root ./dataset/ZuCo \
  --save_path ./checkpoints/decoding \
  --stage1_epochs 20 \
  --stage2_epochs 30 \
  -cuda cuda:0
```

直接评估：

```bash
python eval_decoding.py \
  --checkpoint_path checkpoints/decoding/best/<checkpoint>.pt \
  --config_path config/decoding/<config>.json \
  --data_root ./dataset/ZuCo \
  --test_input EEG \
  --train_input EEG \
  -cuda cuda:0
```

## 说明

- `data.py` 使用 Pearson 相关系数从句子级 EEG 构建功能连接图。
- `model_decoding.py` 冻结文本编码器和 BART 模型，并根据训练阶段控制可训练模块。
- `eval_decoding.py` 基于图 token encoder outputs 调用 BART 生成接口，避免在评估中使用 teacher forcing。
- 默认最大目标文本长度为 `56`。

## 引用

本项目遵循 ZuCo 数据集上的 EEG-to-text 解码设定，并使用生成式评估流程。如果你在实验中使用本代码，请根据具体实验引用相关 EEG-to-text 论文和 ZuCo 数据集。 
