# EEG-To-Text with Graph Memory

本仓库实现了一个面向 ZuCo 数据集的 EEG 到文本解码流程。核心思想是：先把句子级 EEG 构造成电极图，经过图编码器得到 EEG graph embedding，再把图嵌入投影成 BART decoder cross-attention 的 memory tokens。BART 会从这些 memory tokens 中构造 cross-attention 的 key/value，因此生成阶段等价于让冻结的 BART decoder 从 EEG 图记忆中读取信息并生成文本。

评估阶段使用 `model.generate(...)` 自回归生成文本，不使用 teacher forcing 下的 logits 直接解码。

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
│   ├── prepare_dataset.sh
│   ├── prepare_dataset_windows.ps1
│   ├── train_gaet.sh
│   └── eval_gaet.sh
└── util/
    ├── construct_dataset_mat_to_pickle_v1.py
    ├── construct_dataset_mat_to_pickle_v2.py
    └── data_loading_helpers_modified.py
```

## 环境配置

```bash
conda env create -f environment.yml
conda activate EEG
```

## 数据准备

将 ZuCo 的 `.mat` 文件放到以下目录结构中：

```text
dataset/ZuCo/
├── task1-SR/Matlab_files/
├── task2-NR/Matlab_files/
├── task3-TSR/Matlab_files/
└── task2-NR-2.0/Matlab_files/
```

转换为 pickle：

```bash
bash scripts/prepare_dataset.sh
```

Windows PowerShell：

```powershell
.\scripts\prepare_dataset_windows.ps1
```

## 模型思想

1. `data.py` 将每个句子的 EEG 表示为 `(105, num_bands)`，其中 105 个电极是图节点，频段特征是每个节点的信号。
2. 根据 EEG 节点信号计算 Pearson 功能连接图，并进行归一化。
3. `TemporalEncoder + EEGGraphTransformer` 编码 EEG 图，得到 graph embedding。
4. `GraphToLLMProjector` 把 graph embedding 投影成多个 graph memory tokens。
5. 这些 graph memory tokens 作为 `encoder_outputs` 传给 BART。BART decoder 的 cross-attention 会从它们构造 K/V，并在生成每个 token 时读取 EEG 图信息。

## 训练

```bash
bash scripts/train_gaet.sh task1_task2_task3 EEG 0
```

训练分为两个阶段：

1. 对齐阶段：训练 EEG 时序编码器、图 Transformer 和图文对比学习头，让 EEG graph embedding 靠近 frozen text embedding。
2. 生成阶段：冻结 BART 和文本编码器，训练 EEG encoder、graph transformer 和 graph-to-memory projector。若设置 `--contrastive_weight > 0`，冻结的对比头会作为 stage-1 对齐锚点，为 EEG graph embedding 提供额外正则。

checkpoint 默认保存在：

```text
checkpoints/decoding/best/
checkpoints/decoding/last/
```

## 评估

```bash
bash scripts/eval_gaet.sh task1_task2_task3 EEG EEG 0
```

生成文本保存在：

```text
results/
```

指标结果追加写入：

```text
score_results/
```

评估脚本会计算 BLEU、SacreBLEU、WER、CER 和 ROUGE。

## 数据划分

`data.py` 使用规范化后的句子文本构建全局 train/dev/test split，再将所有选中 task 和 subject 的样本映射到相同 split，避免同一句文本跨集合泄漏。

当前模型只使用句子级 EEG 构图；word-level 数据是可选附加字段。数据转换脚本会保留缺少词划分或词级 EEG 的句子，只要它仍包含 `content` 和 `sentence_level_EEG`。

## 直接运行

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

```bash
python eval_decoding.py \
  --checkpoint_path checkpoints/decoding/best/<checkpoint>.pt \
  --config_path config/decoding/<config>.json \
  --data_root ./dataset/ZuCo \
  --test_input EEG \
  --train_input EEG \
  -cuda cuda:0
```
