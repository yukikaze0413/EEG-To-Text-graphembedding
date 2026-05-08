# EEG-To-Text 完整链路说明

本文档说明本仓库从 ZuCo 原始数据到 EEG 图记忆驱动文本生成的完整流程。当前方案的核心思想是：先将句子级 EEG 构造成电极图，经过图编码器得到 EEG graph embedding，再将图嵌入投影成 BART decoder cross-attention 的 memory tokens。BART decoder 会从这些 memory tokens 中构造 cross-attention 的 key/value，从而在生成文本时读取 EEG 图信息。

## 1. 总体流程

完整链路如下：

```text
ZuCo .mat
  -> pickle 数据集
  -> ZuCo_dataset 样本
  -> 句子级 EEG: (105, num_bands)
  -> Pearson 功能连接图: (105, 105)
  -> TemporalEncoder
  -> EEGGraphTransformer
  -> EEG graph embedding
  -> GraphToLLMProjector
  -> graph memory tokens
  -> BART encoder_outputs
  -> BART decoder cross-attention K/V
  -> 自回归文本生成
```

该链路只依赖句子级 EEG 和句子文本。word-level 数据是可选附加字段；如果 ZuCo 中某些样本没有词划分或词级 EEG，只要仍包含 `content` 和 `sentence_level_EEG`，数据转换脚本会保留该句子样本。

## 2. 数据准备

原始 ZuCo 数据以 `.mat` 文件形式存放，目录结构约定为：

```text
dataset/ZuCo/
├── task1-SR/Matlab_files/
├── task2-NR/Matlab_files/
├── task3-TSR/Matlab_files/
└── task2-NR-2.0/Matlab_files/
```

转换脚本：

```bash
bash scripts/prepare_dataset.sh
```

Windows PowerShell：

```powershell
.\scripts\prepare_dataset_windows.ps1
```

转换后生成：

```text
dataset/ZuCo/<task-name>/pickle/<task-name>-dataset.pickle
```

相关文件：

- `util/construct_dataset_mat_to_pickle_v1.py`
- `util/construct_dataset_mat_to_pickle_v2.py`
- `util/data_loading_helpers_modified.py`

转换后的每个句子样本至少包含：

```python
{
    "content": "...",
    "sentence_level_EEG": {
        "mean_t1": ...,
        "mean_t2": ...,
        "mean_a1": ...,
        "mean_a2": ...,
        "mean_b1": ...,
        "mean_b2": ...,
        "mean_g1": ...,
        "mean_g2": ...
    },
    "word": [],
    "word_tokens_has_fixation": [],
    "word_tokens_with_mask": [],
    "word_tokens_all": []
}
```

其中 `word*` 字段可以为空，不影响当前训练链路。

## 3. Dataset 构建

入口文件是 `data.py` 中的 `ZuCo_dataset`。

训练和评估脚本先根据 `task_name` 加载一个或多个 pickle 数据集：

- `task1` -> `task1-SR`
- `task2` -> `task2-NR`
- `task3` -> `task3-TSR`
- `taskNRv2` -> `task2-NR-2.0`

`ZuCo_dataset` 会做三件事：

1. 根据规范化后的句子文本构建全局 train/dev/test split。
2. 将所有选中 task 和 subject 的样本映射到同一个 split。
3. 将句子级 EEG 和文本转换成模型输入。

全局文本级 split 的目的是避免同一句文本因为 subject 或 task 不同而跨集合泄漏。

## 4. EEG 句子级输入表示

每个 ZuCo 频段特征是一个 105 通道向量，对应 105 个 EEG 电极。

`data.py::_get_sent_eeg(...)` 将多个频段 stack 成：

```text
(channels, bands) = (105, num_bands)
```

也就是说：

- 每一行是一个电极节点。
- 每一列是该电极在某个频段上的句子级特征。

这里必须使用 `np.stack(..., axis=1)`。如果先 `concatenate` 再 reshape，会混淆 channel 和 band 的语义，使图节点特征错位。

默认频段为：

```text
_t1, _t2, _a1, _a2, _b1, _b2, _g1, _g2
```

样本输出结构为：

```python
{
    "eeg_signals": Tensor[105, num_bands],
    "adjacency": Tensor[105, 105],
    "target_ids": Tensor[max_len],
    "target_mask": Tensor[max_len],
    "raw_text": str
}
```

## 5. EEG 建图

当前图结构使用句子级 EEG 节点信号构建功能连接图。

步骤如下：

1. 对 `eeg_signals` 计算 Pearson correlation：

```python
a = np.corrcoef(eeg_signals)
```

2. 将 NaN 和 inf 置零。
3. 将负相关裁剪为 0，仅保留非负功能连接。
4. 对角线设为 1，表示 self-loop。
5. 使用对称归一化：

```text
A_norm = D^(-1/2) A D^(-1/2)
```

得到：

```text
adjacency: (105, 105)
```

相关函数：

- `build_functional_adjacency`
- `combine_adjacency_matrices`

## 6. 模型结构

模型定义在 `model_decoding.py`，核心类为 `GAET`。

### 6.1 TemporalEncoder

输入：

```text
eeg_signals: (B, C, T)
```

其中：

- `B` 是 batch size。
- `C=105` 是电极节点数。
- `T=num_bands` 是每个节点的频段信号长度。

`TemporalEncoder` 对每个电极节点的频段序列做 1D convolution 编码，输出：

```text
node_features: (B, C, d_model)
```

### 6.2 EEGGraphTransformer

`EEGGraphTransformer` 接收：

```text
node_features: (B, 105, d_model)
adjacency: (B, 105, 105)
```

其内部使用：

- Laplacian positional encoding
- adjacency 作为 attention bias
- 多层 graph-aware Transformer

输出：

```text
h_graph: (B, d_graph)
h_nodes: (B, 105, d_model)
```

`h_graph` 是整句 EEG 图的全局表示。

### 6.3 GraphToLLMProjector

`GraphToLLMProjector` 将一个 EEG graph embedding 投影成多个 graph memory tokens：

```text
h_graph: (B, d_graph)
graph_memory: (B, num_query_tokens, bart_d_model)
```

这些 tokens 不是普通文本 token，而是 EEG 图记忆。它们会被传入 BART 的 `encoder_outputs`。

## 7. 图嵌入到 BART K/V 的方式

BART 是 encoder-decoder 架构。decoder 每一层都有 cross-attention，其 key/value 来自 encoder hidden states。

本仓库不直接手写每一层的 `past_key_values`，而是使用 HuggingFace BART 的标准接口：

```python
encoder_outputs = BaseModelOutput(last_hidden_state=graph_memory)
out = self.llm(
    encoder_outputs=encoder_outputs,
    attention_mask=encoder_attention_mask,
    labels=labels,
    return_dict=True,
)
```

这样 BART decoder 在 cross-attention 中会自动从 `graph_memory` 计算 K/V。

因此，从机制上看：

```text
EEG graph embedding
  -> graph memory tokens
  -> BART encoder_outputs
  -> decoder cross-attention K/V
  -> generated text
```

这就是“脑电建图后做图嵌入到模型 K/V 中”的当前实现方式。它比手动改 `past_key_values` 更稳定，也更符合 BART 的生成 API。

## 8. 两阶段训练

训练入口是 `train_decoding.py`。

### 8.1 Stage 1: 图文对齐

阶段名：

```text
align
```

目标：

```text
让 EEG graph embedding 和文本 embedding 对齐
```

流程：

1. EEG 经过 temporal encoder 和 graph transformer，得到 `h_graph`。
2. 句子文本经过冻结的 text encoder，得到 `h_text`。
3. 使用 symmetric InfoNCE：

```text
loss = 0.5 * (CE(graph->text) + CE(text->graph))
```

训练模块：

- `TemporalEncoder`
- `EEGGraphTransformer`
- `GraphTextContrastive`

冻结模块：

- text encoder
- BART
- graph-to-memory projector

### 8.2 Stage 2: 图记忆条件生成

阶段名：

```text
generate
```

目标：

```text
把 EEG graph embedding 转成 BART decoder 可读取的 cross-attention memory，并生成目标文本
```

流程：

1. EEG graph encoder 得到 `h_graph`。
2. projector 得到 `graph_memory`。
3. `graph_memory` 作为 BART `encoder_outputs`。
4. BART decoder 通过 cross-attention 从 graph memory 中读取 K/V。
5. 使用目标文本 token 计算生成损失。

训练模块：

- `TemporalEncoder`
- `EEGGraphTransformer`
- `GraphToLLMProjector`

冻结模块：

- BART
- text encoder
- `GraphTextContrastive`

如果 `--contrastive_weight > 0`，会额外计算 stage-1 的 contrastive loss。此时 contrastive head 是冻结的，它作为固定对齐锚点约束 EEG graph embedding，而不是自己漂移来降低 loss。

## 9. Checkpoint 和配置

训练脚本会保存：

```text
checkpoints/decoding/best/
checkpoints/decoding/last/
```

配置保存到：

```text
config/decoding/
```

checkpoint 文件名包含：

- task name
- batch size
- stage1/stage2 epoch 数
- learning rate
- train input 类型

## 10. 评估流程

评估入口是 `eval_decoding.py`。

流程：

1. 读取训练配置。
2. 按训练配置重建模型结构。
3. 加载 checkpoint。
4. 构建 test split。
5. 调用：

```python
model.generate(
    eeg_signals=batch["eeg_signals"],
    adjacency=batch["adjacency"],
    max_length=max_len,
    num_beams=args["num_beams"],
    do_sample=args["do_sample"],
)
```

`generate` 内部同样会：

```text
EEG -> graph embedding -> graph memory -> BART encoder_outputs -> decoder K/V -> generated ids
```

输出：

```text
results/
score_results/
```

指标：

- BLEU
- SacreBLEU
- WER
- CER
- ROUGE

## 11. 噪声基线

`train_input` 或 `test_input` 可以设置为：

```text
EEG
noise
```

当输入为 `noise` 时，`data.py` 会用随机噪声替代真实 EEG：

```python
eeg_signals = torch.randn_like(eeg_signals)
```

这用于评估模型是否真的依赖 EEG 图信息，而不是只学到语言先验或数据偏置。

## 12. 关键设计注意点

### 12.1 不依赖 word-level 数据

当前模型只使用句子级 EEG 图：

```text
content + sentence_level_EEG
```

因此没有词划分的样本仍然可用。转换脚本会保留这些样本，并将 word 字段置空。

### 12.2 channel 和 band 不能混淆

EEG 输入必须是：

```text
(105, num_bands)
```

其中 105 是电极节点。错误的 reshape 会让一个节点混入其他节点的频段值，破坏图结构语义。

### 12.3 BART 的 K/V 来源

本项目通过 `encoder_outputs=graph_memory` 将 EEG 图记忆接入 BART。BART decoder 自己负责从 encoder hidden states 构造 cross-attention K/V。

这与 prefix-tuning 中手动构造每层 `past_key_values` 不同，但实现上更简单稳定，并且语义上仍是“图嵌入作为 decoder cross-attention 的 K/V memory”。

### 12.4 数据划分按文本去重

ZuCo 中同一句文本可能出现在不同 subject 或 task 中。为了避免泄漏，split 以规范化后的句子文本为单位。

## 13. 常用命令

准备数据：

```bash
bash scripts/prepare_dataset.sh
```

训练：

```bash
bash scripts/train_gaet.sh task1_task2_task3 EEG 0
```

评估：

```bash
bash scripts/eval_gaet.sh task1_task2_task3 EEG EEG 0
```

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
