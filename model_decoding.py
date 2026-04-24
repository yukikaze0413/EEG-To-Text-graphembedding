import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, BartForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput


class TemporalEncoder(nn.Module):
    """Encode per-channel EEG temporal signal."""

    def __init__(
        self,
        in_channels: int = 1,
        hidden_dim: int = 64,
        out_dim: int = 256,
        kernel_sizes: Optional[List[int]] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [3, 3, 3]

        layers: List[nn.Module] = []
        cur_in = in_channels
        cur_hidden = hidden_dim
        for k in kernel_sizes:
            layers.extend(
                [
                    nn.Conv1d(cur_in, cur_hidden, kernel_size=k, padding=k // 2),
                    nn.BatchNorm1d(cur_hidden),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            cur_in = cur_hidden
            cur_hidden *= 2

        self.conv_blocks = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(cur_hidden // 2, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        bsz, channels, timesteps = x.shape
        x = x.view(bsz * channels, 1, timesteps)
        x = self.conv_blocks(x)
        x = self.pool(x).squeeze(-1)
        x = self.proj(x)
        return x.view(bsz, channels, -1)


class GraphTransformerLayer(nn.Module):
    """Graph-aware Transformer layer using adjacency as attention bias."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        bsz, nodes, dim = x.shape

        q = self.q_proj(x).view(bsz, nodes, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, nodes, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, nodes, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores + adjacency.unsqueeze(1)

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(bsz, nodes, dim)
        out = self.out_proj(out)

        x = self.norm1(x + self.dropout(out))
        x = self.norm2(x + self.ffn(x))
        return x


class GraphPositionalEncoding(nn.Module):
    """Laplacian positional encoding."""

    def __init__(self, d_model: int, k_eigen: int = 8) -> None:
        super().__init__()
        self.k_eigen = k_eigen
        self.proj = nn.Linear(k_eigen, d_model)

    def forward(self, adjacency: torch.Tensor) -> torch.Tensor:
        bsz, channels, _ = adjacency.shape
        pe_list: List[torch.Tensor] = []
        for i in range(bsz):
            a = adjacency[i].detach().float().cpu().numpy()
            d = np.diag(a.sum(axis=1))
            lap = d - a
            _, eigvecs = np.linalg.eigh(lap)
            eig = eigvecs[:, 1 : self.k_eigen + 1]
            pe = torch.from_numpy(eig).float()
            if pe.shape[1] < self.k_eigen:
                pad = torch.zeros(channels, self.k_eigen - pe.shape[1], dtype=pe.dtype)
                pe = torch.cat([pe, pad], dim=1)
            pe_list.append(pe)
        pe_tensor = torch.stack(pe_list, dim=0).to(adjacency.device)
        return self.proj(pe_tensor)


class EEGGraphTransformer(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        readout: str = "mean",
    ) -> None:
        super().__init__()
        self.readout = readout
        self.pos_enc = GraphPositionalEncoding(d_model)
        self.layers = nn.ModuleList(
            [GraphTransformerLayer(d_model=d_model, num_heads=num_heads, dropout=dropout) for _ in range(num_layers)]
        )

    def forward(self, node_features: torch.Tensor, adjacency: torch.Tensor, return_all: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        x = node_features + self.pos_enc(adjacency)
        for layer in self.layers:
            x = layer(x, adjacency)

        if self.readout == "mean":
            h_graph = x.mean(dim=1)
        elif self.readout == "max":
            h_graph = x.max(dim=1)[0]
        elif self.readout == "meanmax":
            h_graph = torch.cat([x.mean(dim=1), x.max(dim=1)[0]], dim=-1)
        else:
            raise ValueError(f"Unsupported readout: {self.readout}")

        if return_all:
            return h_graph, x
        return h_graph, x


class GraphTextContrastive(nn.Module):
    """Symmetric InfoNCE for graph-text alignment."""

    def __init__(self, d_graph: int = 256, d_text: int = 768, d_proj: int = 256, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature
        self.graph_proj = nn.Sequential(nn.Linear(d_graph, d_proj), nn.ReLU(), nn.Linear(d_proj, d_proj))
        self.text_proj = nn.Sequential(nn.Linear(d_text, d_proj), nn.ReLU(), nn.Linear(d_proj, d_proj))

    def forward(self, h_graph: torch.Tensor, h_text: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z_graph = F.normalize(self.graph_proj(h_graph), dim=-1)
        z_text = F.normalize(self.text_proj(h_text), dim=-1)
        logits = (z_graph @ z_text.t()) / self.temperature
        labels = torch.arange(logits.shape[0], device=logits.device)
        loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))
        return loss, logits


class GraphToLLMProjector(nn.Module):
    """Project graph embedding to graph tokens for decoder cross-attention."""

    def __init__(self, d_graph: int = 256, d_llm: int = 1024, num_query_tokens: int = 32) -> None:
        super().__init__()
        self.num_query_tokens = num_query_tokens
        self.d_llm = d_llm
        self.graph_to_tokens = nn.Sequential(
            nn.Linear(d_graph, d_llm * num_query_tokens // 2),
            nn.GELU(),
            nn.Linear(d_llm * num_query_tokens // 2, d_llm * num_query_tokens),
        )
        self.query_tokens = nn.Parameter(torch.randn(1, num_query_tokens, d_llm) * 0.02)
        self.token_norm = nn.LayerNorm(d_llm)

    def forward(self, h_graph: torch.Tensor) -> torch.Tensor:
        bsz = h_graph.shape[0]
        kv_tokens = self.graph_to_tokens(h_graph).view(bsz, self.num_query_tokens, self.d_llm)
        q = self.query_tokens.expand(bsz, -1, -1)
        return self.token_norm(kv_tokens + q)


class EEGTextEncoder(nn.Module):
    """Frozen text encoder used in stage-1 alignment."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", max_length: int = 128) -> None:
        super().__init__()
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

    def forward(self, text_list: List[str], device: torch.device) -> torch.Tensor:
        with torch.no_grad():
            self.model.eval()
            toks = self.tokenizer(
                text_list,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            toks = {k: v.to(device) for k, v in toks.items()}
            outs = self.model(**toks)
            attn = toks["attention_mask"].unsqueeze(-1).to(outs.last_hidden_state.dtype)
            pooled = (outs.last_hidden_state * attn).sum(dim=1) / torch.clamp(attn.sum(dim=1), min=1.0)
            return pooled


class GAET(nn.Module):
    """
    Graph-Aligned EEG-to-Text:
    Stage-1 contrastive alignment, Stage-2 graph-token conditioned generation.
    """

    def __init__(
        self,
        temporal_cfg: Optional[Dict] = None,
        graph_cfg: Optional[Dict] = None,
        contrastive_cfg: Optional[Dict] = None,
        llm_name: str = "facebook/bart-large",
        num_query_tokens: int = 32,
        stage: str = "align",
    ) -> None:
        super().__init__()
        temporal_cfg = temporal_cfg or {}
        graph_cfg = graph_cfg or {}
        contrastive_cfg = contrastive_cfg or {}

        d_model = graph_cfg.get("d_model", 256)
        readout = graph_cfg.get("readout", "mean")
        if readout == "meanmax":
            d_graph = d_model * 2
        else:
            d_graph = d_model

        temporal_cfg = {**temporal_cfg, "out_dim": d_model}
        self.temporal_encoder = TemporalEncoder(**temporal_cfg)
        self.graph_transformer = EEGGraphTransformer(**graph_cfg)

        self.text_encoder = EEGTextEncoder(model_name=contrastive_cfg.get("text_model", "sentence-transformers/all-MiniLM-L6-v2"))
        self.contrastive = GraphTextContrastive(
            d_graph=d_graph,
            d_text=contrastive_cfg.get("d_text", 384),
            d_proj=contrastive_cfg.get("d_proj", 256),
            temperature=contrastive_cfg.get("temperature", 0.07),
        )

        self.llm = BartForConditionalGeneration.from_pretrained(llm_name)
        for p in self.llm.parameters():
            p.requires_grad = False

        self.projector = GraphToLLMProjector(
            d_graph=d_graph,
            d_llm=self.llm.config.d_model,
            num_query_tokens=num_query_tokens,
        )
        self.stage = stage
        self.set_stage(stage)

    def set_stage(self, stage: str) -> None:
        if stage not in ("align", "generate"):
            raise ValueError("stage must be one of: align, generate")
        self.stage = stage

        for p in self.temporal_encoder.parameters():
            p.requires_grad = stage == "align"
        for p in self.graph_transformer.parameters():
            p.requires_grad = stage == "align"
        for p in self.contrastive.parameters():
            p.requires_grad = stage == "align"
        for p in self.projector.parameters():
            p.requires_grad = stage == "generate"

        # LLM and text encoder stay frozen in both stages.
        for p in self.llm.parameters():
            p.requires_grad = False
        for p in self.text_encoder.parameters():
            p.requires_grad = False
        self.text_encoder.eval()

    def encode_eeg(self, eeg_signals: torch.Tensor, adjacency: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        node_features = self.temporal_encoder(eeg_signals)
        h_graph, h_nodes = self.graph_transformer(node_features, adjacency, return_all=True)
        return h_graph, h_nodes

    def forward_align(self, eeg_signals: torch.Tensor, adjacency: torch.Tensor, text_list: List[str]) -> Dict[str, torch.Tensor]:
        h_graph, _ = self.encode_eeg(eeg_signals, adjacency)
        h_text = self.text_encoder(text_list, eeg_signals.device)
        loss, logits = self.contrastive(h_graph, h_text)
        return {"loss": loss, "logits": logits}

    def _labels_to_model_labels(self, target_ids: torch.Tensor, pad_token_id: int) -> torch.Tensor:
        labels = target_ids.clone()
        labels[labels == pad_token_id] = -100
        return labels

    def forward_generate(
        self,
        eeg_signals: torch.Tensor,
        adjacency: torch.Tensor,
        target_ids: torch.Tensor,
        text_list: Optional[List[str]] = None,
        contrastive_weight: float = 0.0,
    ):
        h_graph, _ = self.encode_eeg(eeg_signals, adjacency)
        graph_tokens = self.projector(h_graph)
        encoder_outputs = BaseModelOutput(last_hidden_state=graph_tokens)
        encoder_attention_mask = torch.ones(
            graph_tokens.shape[0], graph_tokens.shape[1], dtype=torch.long, device=graph_tokens.device
        )
        labels = self._labels_to_model_labels(target_ids, self.llm.config.pad_token_id)
        out = self.llm(
            encoder_outputs=encoder_outputs,
            attention_mask=encoder_attention_mask,
            labels=labels,
            return_dict=True,
        )

        out.contrastive_loss = None
        if contrastive_weight > 0.0 and text_list is not None:
            with torch.no_grad():
                h_text = self.text_encoder(text_list, eeg_signals.device)
            contrastive_loss, _ = self.contrastive(h_graph.detach(), h_text)
            # In generate stage, trainable params are projector-only; this term has no gradient path.
            # Keep it as a monitoring signal and avoid adding a misleading constant to the loss.
            out.contrastive_loss = contrastive_loss
        return out

    def forward(
        self,
        eeg_signals: torch.Tensor,
        adjacency: torch.Tensor,
        target_ids: Optional[torch.Tensor] = None,
        text_list: Optional[List[str]] = None,
        contrastive_weight: float = 0.0,
    ):
        if self.stage == "align":
            if text_list is None:
                raise ValueError("text_list is required in align stage")
            return self.forward_align(eeg_signals, adjacency, text_list)
        if target_ids is None:
            raise ValueError("target_ids is required in generate stage")
        return self.forward_generate(
            eeg_signals=eeg_signals,
            adjacency=adjacency,
            target_ids=target_ids,
            text_list=text_list,
            contrastive_weight=contrastive_weight,
        )

    @torch.no_grad()
    def generate(
        self,
        eeg_signals: torch.Tensor,
        adjacency: torch.Tensor,
        max_length: int = 56,
        num_beams: int = 4,
        do_sample: bool = False,
    ) -> torch.Tensor:
        self.eval()
        h_graph, _ = self.encode_eeg(eeg_signals, adjacency)
        graph_tokens = self.projector(h_graph)
        encoder_outputs = BaseModelOutput(last_hidden_state=graph_tokens)
        encoder_attention_mask = torch.ones(
            graph_tokens.shape[0], graph_tokens.shape[1], dtype=torch.long, device=graph_tokens.device
        )
        return self.llm.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=encoder_attention_mask,
            max_length=max_length,
            num_beams=num_beams,
            do_sample=do_sample,
        )
