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
    np.fill_diagonal(a, 1.0)
    return a.astype(np.float32)


def combine_adjacency_matrices(a_functional: np.ndarray, self_loop_weight: float = 1.0) -> np.ndarray:
    a = a_functional.copy()
    np.fill_diagonal(a, self_loop_weight)
    d = np.sum(a, axis=1)
    d_inv_sqrt = np.diag(1.0 / np.sqrt(d + 1e-8))
    a_norm = d_inv_sqrt @ a @ d_inv_sqrt
    return a_norm.astype(np.float32)


def _get_sent_eeg(sent_obj: Dict, bands: List[str]) -> torch.Tensor:
    sent_eeg_features = []
    for band in bands:
        key = "mean" + band
        sent_eeg_features.append(sent_obj["sentence_level_EEG"][key])
    sent_eeg_embedding = np.concatenate(sent_eeg_features)
    return normalize_1d(torch.from_numpy(sent_eeg_embedding).float())


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

    target_text = sent_obj["content"]
    if "emp11111ty" in target_text:
        target_text = target_text.replace("emp11111ty", "empty")
    if "film.1" in target_text:
        target_text = target_text.replace("film.1", "film.")

    target_tokenized = tokenizer(
        target_text,
        padding="max_length",
        max_length=max_len,
        truncation=True,
        return_tensors="pt",
        return_attention_mask=True,
    )

    sent_level_eeg = _get_sent_eeg(sent_obj, bands)  # (105 * nbands,)
    if torch.isnan(sent_level_eeg).any():
        return None
    num_channels = 105
    num_bands = len(bands)
    if sent_level_eeg.numel() != num_channels * num_bands:
        return None

    eeg_signals = sent_level_eeg.view(num_channels, num_bands).contiguous()  # (C, T)
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

        for dataset_dict in input_dataset_dicts:
            if subject == "ALL":
                subjects = list(dataset_dict.keys())
            else:
                subjects = [subject]

            subject_lengths = [len(dataset_dict[s]) for s in subjects]
            if len(set(subject_lengths)) > 1:
                print(
                    "[WARN] Subject sentence counts are not equal; using min length "
                    f"{min(subject_lengths)} for split consistency: {dict(zip(subjects, subject_lengths))}"
                )
            total_num_sentence = min(subject_lengths)
            train_divider = int(0.8 * total_num_sentence)
            dev_divider = train_divider + int(0.1 * total_num_sentence)

            if setting != "unique_sent":
                raise ValueError("Only unique_sent setting is supported in GAET pipeline.")

            if phase == "train":
                index_range = range(train_divider)
            elif phase == "dev":
                index_range = range(train_divider, dev_divider)
            elif phase == "test":
                index_range = range(dev_divider, total_num_sentence)
            else:
                raise ValueError(f"Unsupported phase: {phase}")

            for key in subjects:
                for i in index_range:
                    sample = get_input_sample(
                        dataset_dict[key][i],
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
