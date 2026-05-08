from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


def normalize_1d(input_tensor: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mean = torch.mean(input_tensor)
    std = torch.std(input_tensor)
    return (input_tensor - mean) / (std + eps)


def build_functional_adjacency(eeg_signals: np.ndarray) -> np.ndarray:
    """
    Build channel functional adjacency using Pearson correlation.
    eeg_signals: (C, T)
    """
    a = np.corrcoef(eeg_signals)
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    a = np.clip(a, 0.0, 1.0)
    np.fill_diagonal(a, 1.0)
    return a.astype(np.float32)


def combine_adjacency_matrices(a_functional: np.ndarray, self_loop_weight: float = 1.0) -> np.ndarray:
    a = a_functional.copy()
    np.fill_diagonal(a, self_loop_weight)
    d = np.sum(a, axis=1)
    d_inv_sqrt = np.diag(1.0 / np.sqrt(d + 1e-8))
    a_norm = d_inv_sqrt @ a @ d_inv_sqrt
    a_norm = np.nan_to_num(a_norm, nan=0.0, posinf=0.0, neginf=0.0)
    return a_norm.astype(np.float32)


def _get_sent_eeg(sent_obj: Dict, bands: List[str]) -> torch.Tensor:
    """
    Return sentence-level EEG as (channels, bands).

    Each ZuCo band feature is stored as a 105-channel vector. Stacking on axis 1
    keeps every row as one electrode node and every column as that node's band
    signal. Concatenating then reshaping would mix channel and band identities.
    """
    sent_eeg_features = []
    for band in bands:
        key = "mean" + band
        band_values = np.asarray(sent_obj["sentence_level_EEG"][key]).squeeze()
        sent_eeg_features.append(band_values)
    sent_eeg_embedding = np.stack(sent_eeg_features, axis=1)
    return normalize_1d(torch.from_numpy(sent_eeg_embedding).float())


def _normalize_text(text: str) -> str:
    if "emp11111ty" in text:
        text = text.replace("emp11111ty", "empty")
    if "film.1" in text:
        text = text.replace("film.1", "film.")
    return text


def _selected_subjects(dataset_dict: Dict, subject: str) -> List[str]:
    if subject == "ALL":
        return list(dataset_dict.keys())
    return [subject]


def _phase_texts(input_dataset_dicts, subject: str, phase: str) -> set:
    """Build one text-level split shared by all selected tasks and subjects."""
    if not isinstance(input_dataset_dicts, list):
        input_dataset_dicts = [input_dataset_dicts]

    ordered_texts: List[str] = []
    seen = set()
    for dataset_dict in input_dataset_dicts:
        for key in _selected_subjects(dataset_dict, subject):
            for sent_obj in dataset_dict[key]:
                if sent_obj is None:
                    continue
                text = _normalize_text(sent_obj["content"])
                if text not in seen:
                    seen.add(text)
                    ordered_texts.append(text)

    train_divider = int(0.8 * len(ordered_texts))
    dev_divider = train_divider + int(0.1 * len(ordered_texts))
    if phase == "train":
        return set(ordered_texts[:train_divider])
    if phase == "dev":
        return set(ordered_texts[train_divider:dev_divider])
    if phase == "test":
        return set(ordered_texts[dev_divider:])
    raise ValueError(f"Unsupported phase: {phase}")


def get_input_sample(
    sent_obj: Optional[Dict],
    tokenizer,
    bands: Optional[List[str]] = None,
    max_len: int = 56,
    test_input: str = "EEG",
) -> Optional[Dict]:
    if bands is None:
        bands = ["_t1", "_t2", "_a1", "_a2", "_b1", "_b2", "_g1", "_g2"]
    if sent_obj is None:
        return None

    target_text = _normalize_text(sent_obj["content"])

    target_tokenized = tokenizer(
        target_text,
        padding="max_length",
        max_length=max_len,
        truncation=True,
        return_tensors="pt",
        return_attention_mask=True,
    )

    sent_level_eeg = _get_sent_eeg(sent_obj, bands)  # (105, nbands)
    if torch.isnan(sent_level_eeg).any():
        return None
    num_channels = 105
    num_bands = len(bands)
    if sent_level_eeg.shape != (num_channels, num_bands):
        return None

    eeg_signals = sent_level_eeg.contiguous()  # (C, T)
    if test_input.lower() == "noise":
        eeg_signals = torch.randn_like(eeg_signals)

    adjacency = build_functional_adjacency(eeg_signals.cpu().numpy())
    adjacency = combine_adjacency_matrices(adjacency)

    return {
        "eeg_signals": eeg_signals.float(),
        "adjacency": torch.from_numpy(adjacency).float(),
        "target_ids": target_tokenized["input_ids"][0].long(),
        "target_mask": target_tokenized["attention_mask"][0].long(),
        "raw_text": target_text,
    }


class ZuCo_dataset(Dataset):
    def __init__(
        self,
        input_dataset_dicts,
        phase: str,
        tokenizer,
        subject: str = "ALL",
        bands: Optional[List[str]] = None,
        setting: str = "unique_sent",
        test_input: str = "EEG",
        max_len: int = 56,
    ) -> None:
        self.inputs: List[Dict] = []
        self.tokenizer = tokenizer

        if bands is None:
            bands = ["_t1", "_t2", "_a1", "_a2", "_b1", "_b2", "_g1", "_g2"]

        if not isinstance(input_dataset_dicts, list):
            input_dataset_dicts = [input_dataset_dicts]

        split_texts = _phase_texts(input_dataset_dicts, subject, phase)

        for dataset_dict in input_dataset_dicts:
            subjects = _selected_subjects(dataset_dict, subject)

            if setting != "unique_sent":
                raise ValueError("Only unique_sent setting is supported in GAET pipeline.")

            for key in subjects:
                for sent_obj in dataset_dict[key]:
                    if sent_obj is None or _normalize_text(sent_obj["content"]) not in split_texts:
                        continue
                    sample = get_input_sample(
                        sent_obj,
                        tokenizer=self.tokenizer,
                        bands=bands,
                        max_len=max_len,
                        test_input=test_input,
                    )
                    if sample is not None:
                        self.inputs.append(sample)

        if len(self.inputs) == 0:
            raise RuntimeError("No valid samples were created.")

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> Dict:
        return self.inputs[idx]
