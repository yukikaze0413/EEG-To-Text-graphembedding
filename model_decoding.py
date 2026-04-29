import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig, AutoTokenizer, AutoModel
import math
import numpy as np

# 该文件定义 EEG->文本解码相关模型，包括：
# 1) 传统 Transformer 编码分支
# 2) 双向 Mamba 风格编码分支
# 3) EEG-Text 对比对齐分支

""" main architecture for open vocabulary EEG-To-Text decoding"""
class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer for EEG graph processing
    """
    def __init__(self, in_features, out_features, dropout=0.1, alpha=0.2, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.W.weight.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = self.W(h) # (..., num_nodes, out_features)
        
        a_input1 = torch.matmul(Wh, self.a[:self.out_features, :])
        a_input2 = torch.matmul(Wh, self.a[self.out_features:, :])
        
        e = self.leakyrelu(a_input1 + a_input2.transpose(-1, -2))
        e = e + adj

        attention = F.softmax(e, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

class EEGGATEncoder(nn.Module):
    """基于 GAT 的 EEG 编码器，内置自学习动态图邻接矩阵"""
    def __init__(self, in_feature=840, num_nodes=105, bands=8, num_layers=2, dropout=0.1):
        super(EEGGATEncoder, self).__init__()
        self.num_nodes = num_nodes
        self.bands = bands
        assert in_feature == num_nodes * bands, f"in_feature {in_feature} must equal num_nodes {num_nodes} * bands {bands}"
        
        # Learnable adjacency matrix for the graph (自学习动态图)
        # 理论优化：初始化为0，因为 softmax(x+c) = softmax(x)，全为相同常数在初期不起作用。
        # 初始为 0 意味着最初完全依赖数据驱动的动态注意力，随后慢慢学到静态的空间先验偏置。
        self.adj = nn.Parameter(torch.zeros(num_nodes, num_nodes))
        
        # GAT Layers
        self.gat_layers = nn.ModuleList()
        # Initial layer: bands -> 16
        self.gat_layers.append(GraphAttentionLayer(bands, 16, dropout=dropout, concat=True))
        # Hidden layers if any
        for _ in range(num_layers - 2):
            self.gat_layers.append(GraphAttentionLayer(16, 16, dropout=dropout, concat=True))
        # Final layer: 16 -> bands
        self.gat_layers.append(GraphAttentionLayer(16, bands, dropout=dropout, concat=False))
        
    def forward(self, x):
        # x: (batch_size, seq_len, in_feature)
        bs, seq_len, _ = x.shape
        
        # data.py 中拼接顺序是：先按照 band 循环，每个 band 长度为 105
        # 所以一维向量的实际布局是 (bands, num_nodes)
        h = x.view(bs * seq_len, self.bands, self.num_nodes)
        # 转置为 (batch_size * seq_len, num_nodes, bands) 送入 GAT
        h = h.transpose(1, 2)
        
        # GAT passes
        for gat in self.gat_layers:
            h = gat(h, self.adj)
            
        # 恢复原状送给后续的 Transformer 或映射层
        h = h.transpose(1, 2).contiguous()
        out = h.view(bs, seq_len, self.num_nodes * self.bands)
        
        return out



class MambaBlock(nn.Module):
    """轻量级线性复杂度 SSM 风格模块（Mamba inspired）。"""
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        # expand 后的中间通道维度
        inner_dim = d_model * expand
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, inner_dim * 2)
        self.dw_conv = nn.Conv1d(inner_dim, inner_dim, kernel_size=d_conv, padding=d_conv - 1, groups=inner_dim)
        self.act = nn.SiLU()
        self.state_proj = nn.Linear(inner_dim, d_state)
        self.state_out = nn.Linear(d_state, inner_dim)
        self.out_proj = nn.Linear(inner_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 残差连接主分支
        residual = x
        # 先归一化再做投影
        x = self.norm(x)
        # 分成内容分支 u 与门控分支 gate
        u, gate = self.in_proj(x).chunk(2, dim=-1)

        # 深度可分离卷积进行局部时序混合（序列长度线性）
        u_conv = self.dw_conv(u.transpose(1, 2)).transpose(1, 2)
        u_conv = u_conv[:, :u.shape[1], :]
        u_conv = self.act(u_conv)

        # 简化的递推状态扫描（序列长度线性）
        alpha = torch.sigmoid(self.state_proj(u_conv))
        beta = torch.tanh(self.state_proj(u))
        state = torch.zeros(u.shape[0], alpha.shape[-1], device=u.device, dtype=u.dtype)
        states = []
        for t in range(u.shape[1]):
            state = alpha[:, t, :] * state + (1.0 - alpha[:, t, :]) * beta[:, t, :]
            states.append(state)
        s = torch.stack(states, dim=1)
        s = self.state_out(s)

        # 门控融合局部卷积特征与状态特征
        y = (u_conv + s) * torch.sigmoid(gate)
        y = self.out_proj(y)
        y = self.dropout(y)
        # 残差输出
        return residual + y


class BiMambaEEGEncoder(nn.Module):
    """双向 Mamba 编码器：前向/后向分支参数独立。"""
    def __init__(self, d_model, num_layers=6, d_state=16, d_conv=4, expand=2, dropout=0.1, fusion="concat_linear"):
        super().__init__()
        self.fusion = fusion

        # 前向扫描分支
        self.forward_blocks = nn.ModuleList([
            MambaBlock(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand, dropout=dropout)
            for _ in range(num_layers)
        ])
        # 后向扫描分支（输入翻转后处理）
        self.backward_blocks = nn.ModuleList([
            MambaBlock(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand, dropout=dropout)
            for _ in range(num_layers)
        ])

        # 双向融合策略：拼接线性映射 / 门控加权和
        if fusion == "concat_linear":
            self.fusion_proj = nn.Linear(d_model * 2, d_model)
        elif fusion == "gated_sum":
            self.gate_proj = nn.Linear(d_model * 2, d_model)
        else:
            raise ValueError(f"Unsupported fusion type: {fusion}")

    def _run_blocks(self, x, blocks):
        for blk in blocks:
            x = blk(x)
        return x

    def forward(self, x):
        # 前向分支
        x_f = self._run_blocks(x, self.forward_blocks)

        # 后向分支：先翻转时间维，处理后再翻回
        x_rev = torch.flip(x, dims=[1])
        x_b = self._run_blocks(x_rev, self.backward_blocks)
        x_b = torch.flip(x_b, dims=[1])

        if self.fusion == "concat_linear":
            x_out = self.fusion_proj(torch.cat([x_f, x_b], dim=-1))
        else:
            gate = torch.sigmoid(self.gate_proj(torch.cat([x_f, x_b], dim=-1)))
            x_out = gate * x_f + (1.0 - gate) * x_b
        return x_out


class EEGContrastHead(nn.Module):
    # 对比学习头：仅训练 EEG 侧投影，文本编码器冻结
    def __init__(self, in_dim=840, proj_dim=768, temperature=0.07, text_embed_model="sentence-transformers/all-mpnet-base-v2"):
        super().__init__()
        self.eeg_proj = nn.Linear(in_dim, proj_dim)
        self.temperature = temperature

        # 加载并冻结文本句向量模型
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_embed_model)
        self.text_encoder = AutoModel.from_pretrained(text_embed_model)
        for p in self.text_encoder.parameters():
            p.requires_grad = False
        self.text_encoder.eval()

    @staticmethod
    def _masked_mean(x, attn_mask):
        # 按 attention mask 做平均池化
        m = attn_mask.unsqueeze(-1).to(x.dtype)
        x = x * m
        denom = torch.clamp(m.sum(dim=1), min=1.0)
        return x.sum(dim=1) / denom

    def _encode_text(self, text_list, device):
        # 文本侧不参与梯度更新
        with torch.no_grad():
            toks = self.text_tokenizer(
                text_list,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            )
            toks = {k: v.to(device) for k, v in toks.items()}
            outs = self.text_encoder(**toks)
            text_emb = self._masked_mean(outs.last_hidden_state, toks["attention_mask"])
        return text_emb

    def forward(self, eeg_hidden, eeg_attn_mask, text_list):
        # 计算 EEG/文本句向量并进行归一化
        eeg_emb = self._masked_mean(eeg_hidden, eeg_attn_mask)
        eeg_emb = self.eeg_proj(eeg_emb)
        text_emb = self._encode_text(text_list, eeg_hidden.device)

        eeg_emb = F.normalize(eeg_emb, dim=-1)
        text_emb = F.normalize(text_emb, dim=-1)

        # 对称 InfoNCE：EEG->Text 与 Text->EEG
        logits = (eeg_emb @ text_emb.t()) / self.temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        loss_eeg = F.cross_entropy(logits, labels)
        loss_text = F.cross_entropy(logits.t(), labels)
        return 0.5 * (loss_eeg + loss_text)


class BrainTranslator(nn.Module):
    # 主模型：EEG 编码 + 预训练生成模型解码 + 可选对比对齐
    def __init__(
            self,
            pretrained_layers,
            in_feature=840,
            decoder_embedding_size=1024,
            additional_encoder_nhead=8,
            additional_encoder_dim_feedforward=2048,
            eeg_encoder_type="transformer",
            mamba_num_layers=6,
            mamba_d_state=16,
            mamba_d_conv=4,
            mamba_expand=2,
            mamba_dropout=0.1,
            bimamba_fusion="concat_linear",
            gat_num_layers=2,
            gat_dropout=0.1,
            use_contrastive_align=False,
            contrastive_proj_dim=768,
            contrastive_temperature=0.07,
            text_embed_model="sentence-transformers/all-mpnet-base-v2",
    ):
        super(BrainTranslator, self).__init__()

        # 预训练生成模型（BART/Pegasus 等）
        self.pretrained = pretrained_layers
        # EEG 编码器类型
        self.eeg_encoder_type = eeg_encoder_type
        # 是否启用对比对齐
        self.use_contrastive_align = use_contrastive_align

        # 根据配置选择 EEG 编码器
        if eeg_encoder_type == "transformer":
            self.additional_encoder_layer = nn.TransformerEncoderLayer(
                d_model=in_feature,
                nhead=additional_encoder_nhead,
                dim_feedforward=additional_encoder_dim_feedforward,
                batch_first=True,
            )
            self.additional_encoder = nn.TransformerEncoder(self.additional_encoder_layer, num_layers=6)
        elif eeg_encoder_type == "bimamba":
            self.additional_encoder = BiMambaEEGEncoder(
                d_model=in_feature,
                num_layers=mamba_num_layers,
                d_state=mamba_d_state,
                d_conv=mamba_d_conv,
                expand=mamba_expand,
                dropout=mamba_dropout,
                fusion=bimamba_fusion,
            )
        elif eeg_encoder_type == "gat":
            self.additional_encoder = EEGGATEncoder(
                in_feature=in_feature,
                num_nodes=105,
                bands=8,
                num_layers=gat_num_layers,
                dropout=gat_dropout
            )
        else:
            raise ValueError(f"Unsupported eeg_encoder_type: {eeg_encoder_type}")

        # 将 EEG 编码隐状态映射到解码器 embedding 维度
        self.fc1 = nn.Linear(in_feature, decoder_embedding_size)

        # 可选：挂载对比学习头
        if self.use_contrastive_align:
            self.contrast_head = EEGContrastHead(
                in_dim=in_feature,
                proj_dim=contrastive_proj_dim,
                temperature=contrastive_temperature,
                text_embed_model=text_embed_model,
            )

    def addin_forward(self,input_embeddings_batch,  input_masks_invert):
        """input_embeddings_batch: batch_size*Seq_len*840"""
        """input_mask: 1 is not masked, 0 is masked"""
        """input_masks_invert: 1 is masked, 0 is not masked"""

        # Transformer 分支使用 padding mask；Mamba 分支无需该 mask
        if self.eeg_encoder_type == "transformer":
            encoded_hidden = self.additional_encoder(input_embeddings_batch, src_key_padding_mask=input_masks_invert)
        else:
            encoded_hidden = self.additional_encoder(input_embeddings_batch)

        # 投影到生成模型输入空间
        encoded_embedding = F.relu(self.fc1(encoded_hidden))
        return encoded_embedding, encoded_hidden

    @torch.no_grad()
    def generate(
            self,
            input_embeddings_batch, input_masks_batch, input_masks_invert, target_ids_batch_converted,
            generation_config = None,
            logits_processor = None,
            stopping_criteria = None,
            prefix_allowed_tokens_fn= None,
            synced_gpus= None,
            assistant_model = None,
            streamer= None,
            negative_prompt_ids= None,
            negative_prompt_attention_mask = None,
            **kwargs,
    ):
        # 生成阶段仅使用生成分支，不计算对比损失
        encoded_embedding, _ = self.addin_forward(input_embeddings_batch, input_masks_invert)
        output=self.pretrained.generate(
            inputs_embeds = encoded_embedding,
            attention_mask = input_masks_batch[:,:encoded_embedding.shape[1]],
            labels = target_ids_batch_converted,
            return_dict = True,
            generation_config=generation_config,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            synced_gpus=synced_gpus,
            assistant_model=assistant_model,
            streamer=streamer,
            negative_prompt_ids=negative_prompt_ids,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            **kwargs,)

        return output

    def forward(self, input_embeddings_batch, input_masks_batch, input_masks_invert, target_ids_batch_converted, text_list=None, contrastive_weight=0.0):
        # 编码 EEG 输入
        encoded_embedding, encoded_hidden = self.addin_forward(input_embeddings_batch, input_masks_invert)
        # 计算生成损失
        out = self.pretrained(
            inputs_embeds=encoded_embedding,
            attention_mask=input_masks_batch,
            return_dict=True,
            labels=target_ids_batch_converted,
        )

        # 可选叠加对比损失
        contrastive_loss = None
        if self.use_contrastive_align and (text_list is not None) and contrastive_weight > 0.0:
            contrastive_loss = self.contrast_head(encoded_hidden, input_masks_batch, text_list)
            out.loss = out.loss + contrastive_weight * contrastive_loss

        # 附加返回，便于训练日志统计
        out.contrastive_loss = contrastive_loss
        return out


from transformers import T5Tokenizer
"""开放词表 EEG-To-Text 解码主架构"""
class T5Translator(nn.Module):
    def __init__(self, pretrained_layers, in_feature = 840, decoder_embedding_size = 1024, additional_encoder_nhead=8, additional_encoder_dim_feedforward = 2048):
        super(T5Translator, self).__init__()
        
        self.pretrained = pretrained_layers

        self.tokenizer = T5Tokenizer.from_pretrained("t5-large")
        
        # 额外 Transformer 编码器（沿用 BART 方案）
        self.additional_encoder_layer = nn.TransformerEncoderLayer(d_model=in_feature, nhead=additional_encoder_nhead,  dim_feedforward = additional_encoder_dim_feedforward, batch_first=True)
        self.additional_encoder = nn.TransformerEncoder(self.additional_encoder_layer, num_layers=6)
        
        # print('[INFO]添加位置编码')
        # self.positional_embedding = PositionalEncoding(in_feature)

        self.fc1 = nn.Linear(in_feature, decoder_embedding_size)

    def addin_forward(self,input_embeddings_batch,  input_masks_invert):
        """input_embeddings_batch: batch_size*Seq_len*840"""
        """input_mask: 1 is not masked, 0 is masked"""
        """input_masks_invert: 1 is masked, 0 is not masked"""

        # input_embeddings_batch = self.positional_embedding(input_embeddings_batch)
        # 使用 src_key_padding_mask 处理补齐位
        encoded_embedding = self.additional_encoder(input_embeddings_batch, src_key_padding_mask=input_masks_invert)

        # encoded_embedding = self.additional_encoder(input_embeddings_batch)
        encoded_embedding = F.relu(self.fc1(encoded_embedding))
        return encoded_embedding

    @torch.no_grad()
    def generate(
            self,
            input_embeddings_batch, input_masks_batch, input_masks_invert, target_ids_batch_converted,
            generation_config = None,
            logits_processor = None,
            stopping_criteria = None,
            prefix_allowed_tokens_fn= None,
            synced_gpus= None,
            assistant_model = None,
            streamer= None,
            negative_prompt_ids= None,
            negative_prompt_attention_mask = None,
            **kwargs,
    ):
        encoded_embedding=self.addin_forward(input_embeddings_batch, input_masks_invert)


        input_ids = self.tokenizer("transcribe in English: ", return_tensors="pt").input_ids.to(encoded_embedding.device)
        self.task_embedding = self.pretrained.shared(input_ids).to(encoded_embedding.device)
        task_embedding = self.task_embedding.repeat(encoded_embedding.size(0), 1, 1).to(encoded_embedding.device)
        encoded_embedding = torch.cat((task_embedding, encoded_embedding), dim=1)
        input_masks_batch = torch.cat((torch.ones(encoded_embedding.size(0), task_embedding.size(1)).to(encoded_embedding.device), input_masks_batch), dim=1)


        output=self.pretrained.generate(
            inputs_embeds = encoded_embedding,
            attention_mask = input_masks_batch[:,:encoded_embedding.shape[1]],
            labels = target_ids_batch_converted,
            return_dict = True,
            generation_config=generation_config,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            synced_gpus=synced_gpus,
            assistant_model=assistant_model,
            streamer=streamer,
            negative_prompt_ids=negative_prompt_ids,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            **kwargs,)

        return output

    def forward(self, input_embeddings_batch, input_masks_batch, input_masks_invert, target_ids_batch_converted):
        encoded_embedding=self.addin_forward(input_embeddings_batch, input_masks_invert)
        
        # 任务前缀定义
        input_ids = self.tokenizer("transcribe in English: ", return_tensors="pt").input_ids.to(encoded_embedding.device)
        self.task_embedding = self.pretrained.shared(input_ids).to(encoded_embedding.device)
        task_embedding = self.task_embedding.repeat(encoded_embedding.size(0), 1, 1).to(encoded_embedding.device)
        encoded_embedding = torch.cat((task_embedding, encoded_embedding), dim=1)
        input_masks_batch = torch.cat((torch.ones(encoded_embedding.size(0), task_embedding.size(1)).to(encoded_embedding.device), input_masks_batch), dim=1)

        out = self.pretrained(inputs_embeds = encoded_embedding, attention_mask = input_masks_batch,
                              return_dict = True, labels = target_ids_batch_converted)
        return out


"""简化版开放词表 EEG-To-Text 解码模型（不含额外 MTE 编码器）"""
class BrainTranslatorNaive(nn.Module):
    def __init__(self, pretrained_layers, in_feature = 840, decoder_embedding_size = 1024, additional_encoder_nhead=8, additional_encoder_dim_feedforward = 2048):
        super(BrainTranslatorNaive, self).__init__()
        '''不使用额外 Transformer 编码器的版本'''
        self.pretrained = pretrained_layers
        self.fc1 = nn.Linear(in_feature, decoder_embedding_size)

    def forward(self, input_embeddings_batch, input_masks_batch, input_masks_invert, target_ids_batch_converted):
        """input_embeddings_batch: batch_size*Seq_len*840"""
        """input_mask: 1 is not masked, 0 is masked"""
        """input_masks_invert: 1 is masked, 0 is not masked"""
        encoded_embedding = F.relu(self.fc1(input_embeddings_batch))
        out = self.pretrained(inputs_embeds = encoded_embedding, attention_mask = input_masks_batch, return_dict = True, labels = target_ids_batch_converted)                    
        return out


"""辅助模块"""
# 基于 BertPooler 修改
class Pooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # 通过取首个 token 的隐状态进行池化
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

# 参考：https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # print('[DEBUG] 输入尺寸:', x.size())
        # print('[DEBUG] 位置编码尺寸:', self.pe.size())
        x = x + self.pe[:x.size(0), :]
        # print('[DEBUG] 加位置编码后的尺寸:', x.size())
        return self.dropout(x)


"""其他实验模块（当前效果一般）"""
class BrainTranslatorBert(nn.Module):
    def __init__(self, pretrained_layers, in_feature = 840, hidden_size = 768):
        super(BrainTranslatorBert, self).__init__()

        self.pretrained_Bert = pretrained_layers
        self.fc1 = nn.Linear(in_feature, hidden_size)

    def forward(self, input_embeddings_batch, input_masks_batch, target_ids_batch):
        embedding = F.relu(self.fc1(input_embeddings_batch))
        out = self.pretrained_Bert(inputs_embeds = embedding, attention_mask = input_masks_batch, labels = target_ids_batch, return_dict = True)
        return out

class EEG2BertMapping(nn.Module):
    def __init__(self, in_feature = 840, hidden_size = 512, out_feature = 768):
        super(EEG2BertMapping, self).__init__()
        self.fc1 = nn.Linear(in_feature, hidden_size)
        self.fc2 = nn.Linear(hidden_size, out_feature)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = self.fc2(out)
        return out

class ContrastiveBrainTextEncoder(nn.Module):
    def __init__(self, pretrained_text_encoder, in_feature = 840, eeg_encoder_nhead=8, eeg_encoder_dim_feedforward = 2048, embed_dim = 768):
        super(ContrastiveBrainTextEncoder, self).__init__()
        # EEG 编码器
        self.positional_embedding = PositionalEncoding(in_feature)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=in_feature, nhead=eeg_encoder_nhead,  dim_feedforward = eeg_encoder_dim_feedforward, batch_first=True)
        self.EEG_Encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)
        self.EEG_pooler = Pooler(in_feature)
        self.ln_final = nn.LayerNorm(in_feature) # 该层可继续调优
        
        # 投影到文本嵌入空间
        self.EEG_projection = nn.Parameter(torch.empty(in_feature, embed_dim))
        
        # 文本编码器
        self.TextEncoder = pretrained_text_encoder
        
        # 可学习温度参数
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, input_EEG_features, input_EEG_attn_mask, input_ids, input_text_attention_masks):
        # 添加位置编码
        input_EEG_features = self.positional_embedding(input_EEG_features)
        # 提取 EEG 特征嵌入
        EEG_hiddenstates = self.EEG_Encoder(input_EEG_features,  src_key_padding_mask = input_EEG_attn_mask)
        EEG_hiddenstates = self.ln_final(EEG_hiddenstates)
        EEG_features = self.EEG_pooler(EEG_hiddenstates) # [N, 840]

        # 映射到文本嵌入维度
        EEG_features = EEG_features @ self.EEG_projection # [N, 768]

        # 提取文本特征嵌入
        Text_features = self.TextEncoder(input_ids = input_ids, attention_mask = input_text_attention_masks, return_dict = True).pooler_output # [N, 768]
        
        # 特征归一化
        EEG_features = EEG_features / EEG_features.norm(dim=-1, keepdim=True) # [N, 768]
        Text_features = Text_features / Text_features.norm(dim=-1, keepdim=True) # [N, 768]

        # 使用余弦相似度构造 logits
        logit_scale = self.logit_scale.exp() 
        logits_per_EEG = logit_scale * EEG_features @ Text_features.t() # [N, N]
        logits_per_text = logit_scale * Text_features @ EEG_features.t() # [N, N]

        return logits_per_EEG, logits_per_text
