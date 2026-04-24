import copy
import json
import os
import pickle
import time
from typing import Dict, Tuple

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BartTokenizer

from config import get_config
from data import ZuCo_dataset
from model_decoding import GAET


def _load_task_dataset(data_root: str, folder: str, source_name: str, pickle_name: str):
    candidates = [
        os.path.join(data_root, folder, "pickle", pickle_name),
        os.path.join(data_root, folder, source_name),
    ]
    for path in candidates:
        if os.path.exists(path):
            with open(path, "rb") as handle:
                print(f"[INFO] loading dataset: {path}")
                return pickle.load(handle)
    raise FileNotFoundError(f"Could not find dataset file. Tried: {candidates}")


def build_dataset(task_name: str, data_root: str):
    whole_dataset_dicts = []
    if "task1" in task_name:
        whole_dataset_dicts.append(
            _load_task_dataset(
                data_root=data_root,
                folder="task1-SR",
                source_name="task1_source.pkl",
                pickle_name="task1-SR-dataset.pickle",
            )
        )
    if "task2" in task_name:
        whole_dataset_dicts.append(
            _load_task_dataset(
                data_root=data_root,
                folder="task2-NR",
                source_name="task2_source.pkl",
                pickle_name="task2-NR-dataset.pickle",
            )
        )
    if "task3" in task_name:
        whole_dataset_dicts.append(
            _load_task_dataset(
                data_root=data_root,
                folder="task3-TSR",
                source_name="task3_source.pkl",
                pickle_name="task3-TSR-dataset.pickle",
            )
        )
    if "taskNRv2" in task_name:
        whole_dataset_dicts.append(
            _load_task_dataset(
                data_root=data_root,
                folder="task2-NR-2.0",
                source_name="taskNRv2_source.pkl",
                pickle_name="task2-NR-2.0-dataset.pickle",
            )
        )
    if len(whole_dataset_dicts) == 0:
        raise RuntimeError(f"No dataset selected by task_name={task_name}")
    return whole_dataset_dicts


def to_device(batch: Dict, device: torch.device) -> Dict:
    return {
        "eeg_signals": batch["eeg_signals"].to(device).float(),
        "adjacency": batch["adjacency"].to(device).float(),
        "target_ids": batch["target_ids"].to(device).long(),
        "target_mask": batch["target_mask"].to(device).long(),
        "raw_text": batch["raw_text"],
    }


def run_epoch(
    model: GAET,
    loader: DataLoader,
    device: torch.device,
    optimizer: AdamW,
    stage: str,
    contrastive_weight: float = 0.0,
    grad_clip: float = 1.0,
    train: bool = True,
) -> Tuple[float, float]:
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_contrastive = 0.0
    total_count = 0

    for batch in tqdm(loader):
        batch = to_device(batch, device)
        bsz = batch["eeg_signals"].size(0)
        total_count += bsz

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            if stage == "align":
                out = model(
                    eeg_signals=batch["eeg_signals"],
                    adjacency=batch["adjacency"],
                    text_list=list(batch["raw_text"]),
                )
                loss = out["loss"]
                contrastive_loss = out["loss"]
            else:
                out = model(
                    eeg_signals=batch["eeg_signals"],
                    adjacency=batch["adjacency"],
                    target_ids=batch["target_ids"],
                    text_list=list(batch["raw_text"]),
                    contrastive_weight=contrastive_weight,
                )
                loss = out.loss
                contrastive_loss = out.contrastive_loss

            if train:
                loss.backward()
                clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), grad_clip)
                optimizer.step()

        total_loss += loss.detach().item() * bsz
        if contrastive_loss is not None:
            total_contrastive += contrastive_loss.detach().item() * bsz

    avg_loss = total_loss / max(1, total_count)
    avg_contrastive = total_contrastive / max(1, total_count)
    return avg_loss, avg_contrastive


def train_stage(
    model: GAET,
    dataloaders: Dict[str, DataLoader],
    device: torch.device,
    stage: str,
    epochs: int,
    lr: float,
    weight_decay: float,
    grad_clip: float,
    checkpoint_path_best: str,
    checkpoint_path_last: str,
    contrastive_weight: float = 0.0,
    contrastive_warmup_epochs: int = 0,
):
    model.set_stage(stage)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, epochs))

    best_dev_loss = float("inf")
    best_wts = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        print(f"[{stage}] Epoch {epoch}/{epochs - 1}")
        if stage == "generate" and epoch >= contrastive_warmup_epochs:
            current_contrastive_weight = contrastive_weight
        else:
            current_contrastive_weight = 0.0

        train_loss, train_ctr = run_epoch(
            model=model,
            loader=dataloaders["train"],
            device=device,
            optimizer=optimizer,
            stage=stage,
            contrastive_weight=current_contrastive_weight,
            grad_clip=grad_clip,
            train=True,
        )
        dev_loss, dev_ctr = run_epoch(
            model=model,
            loader=dataloaders["dev"],
            device=device,
            optimizer=optimizer,
            stage=stage,
            contrastive_weight=current_contrastive_weight,
            grad_clip=grad_clip,
            train=False,
        )
        scheduler.step()

        print(
            f"[{stage}] train_loss={train_loss:.4f} train_ctr={train_ctr:.4f} "
            f"dev_loss={dev_loss:.4f} dev_ctr={dev_ctr:.4f}"
        )

        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            best_wts = copy.deepcopy(model.state_dict())
            torch.save(best_wts, checkpoint_path_best)
            print(f"[{stage}] saved best checkpoint: {checkpoint_path_best}")

    model.load_state_dict(best_wts)
    torch.save(model.state_dict(), checkpoint_path_last)
    print(f"[{stage}] saved last checkpoint: {checkpoint_path_last}")
    return model


if __name__ == "__main__":
    args = get_config("train_decoding")

    seed_val = 312
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    device = torch.device(args["cuda"] if torch.cuda.is_available() else "cpu")
    print(f"[INFO] using device: {device}")

    os.makedirs(args["save_path"], exist_ok=True)
    os.makedirs(os.path.join(args["save_path"], "best"), exist_ok=True)
    os.makedirs(os.path.join(args["save_path"], "last"), exist_ok=True)
    os.makedirs("./config/decoding", exist_ok=True)

    save_name = (
        f"{args['task_name']}_gaet_b{args['batch_size']}_"
        f"s1{args['stage1_epochs']}_s2{args['stage2_epochs']}_"
        f"lr{args['stage1_lr']}_{args['stage2_lr']}_{args['train_input']}"
    )
    ckpt_best = os.path.join(args["save_path"], "best", f"{save_name}.pt")
    ckpt_last = os.path.join(args["save_path"], "last", f"{save_name}.pt")

    with open(os.path.join("./config/decoding", f"{save_name}.json"), "w") as fp:
        json.dump(args, fp, indent=2)

    tokenizer = BartTokenizer.from_pretrained(args["llm_name"])
    datasets = build_dataset(args["task_name"], args["data_root"])

    train_set = ZuCo_dataset(
        datasets,
        phase="train",
        tokenizer=tokenizer,
        subject=args["subjects"],
        bands=args["eeg_bands"],
        setting="unique_sent",
        test_input=args["train_input"],
        max_len=args["max_len"],
    )
    dev_set = ZuCo_dataset(
        datasets,
        phase="dev",
        tokenizer=tokenizer,
        subject=args["subjects"],
        bands=args["eeg_bands"],
        setting="unique_sent",
        test_input=args["train_input"],
        max_len=args["max_len"],
    )
    dataloaders = {
        "train": DataLoader(train_set, batch_size=args["batch_size"], shuffle=True, num_workers=args["num_workers"]),
        "dev": DataLoader(dev_set, batch_size=args["batch_size"], shuffle=False, num_workers=args["num_workers"]),
    }
    print(f"[INFO] train/dev size: {len(train_set)}/{len(dev_set)}")

    temporal_cfg = {
        "hidden_dim": args["temporal_hidden_dim"],
        "kernel_sizes": args["temporal_kernels"],
        "dropout": args["temporal_dropout"],
    }
    graph_cfg = {
        "d_model": args["d_model"],
        "num_layers": args["graph_layers"],
        "num_heads": args["graph_heads"],
        "dropout": args["graph_dropout"],
        "readout": args["graph_readout"],
    }
    contrastive_cfg = {
        "d_text": args["contrastive_d_text"],
        "d_proj": args["contrastive_d_proj"],
        "temperature": args["contrastive_temperature"],
        "text_model": args["text_model"],
    }

    model = GAET(
        temporal_cfg=temporal_cfg,
        graph_cfg=graph_cfg,
        contrastive_cfg=contrastive_cfg,
        llm_name=args["llm_name"],
        num_query_tokens=args["num_query_tokens"],
        stage="align",
    ).to(device)

    start_time = time.time()

    if args["stage1_epochs"] > 0:
        model = train_stage(
            model=model,
            dataloaders=dataloaders,
            device=device,
            stage="align",
            epochs=args["stage1_epochs"],
            lr=args["stage1_lr"],
            weight_decay=args["weight_decay"],
            grad_clip=args["gradient_clip"],
            checkpoint_path_best=ckpt_best,
            checkpoint_path_last=ckpt_last,
        )

    if args["stage2_epochs"] > 0:
        model = train_stage(
            model=model,
            dataloaders=dataloaders,
            device=device,
            stage="generate",
            epochs=args["stage2_epochs"],
            lr=args["stage2_lr"],
            weight_decay=args["weight_decay"],
            grad_clip=args["gradient_clip"],
            checkpoint_path_best=ckpt_best,
            checkpoint_path_last=ckpt_last,
            contrastive_weight=args["contrastive_weight"],
            contrastive_warmup_epochs=args["contrastive_warmup_epochs"],
        )

    elapsed = time.time() - start_time
    print(f"[INFO] training finished in {elapsed / 60:.2f} minutes")
