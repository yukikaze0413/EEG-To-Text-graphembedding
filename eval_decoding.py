import json
import os
import pickle
import time
from typing import Dict, List

import evaluate
import numpy as np
import torch
from nltk.translate.bleu_score import corpus_bleu
from rouge import Rouge
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
        "raw_text": batch["raw_text"],
    }


def compute_metrics(pred_texts: List[str], ref_texts: List[str]) -> Dict:
    target_tokens = [[r.split()] for r in ref_texts]
    pred_tokens = [p.split() for p in pred_texts]
    weights_list = [(1.0,), (0.5, 0.5), (1.0 / 3, 1.0 / 3, 1.0 / 3), (0.25, 0.25, 0.25, 0.25)]
    bleu = [corpus_bleu(target_tokens, pred_tokens, weights=w) for w in weights_list]

    sacrebleu = evaluate.load("sacrebleu").compute(
        predictions=pred_texts,
        references=[[x] for x in ref_texts],
    )
    cer = evaluate.load("cer").compute(predictions=pred_texts, references=ref_texts)
    wer = evaluate.load("wer").compute(predictions=pred_texts, references=ref_texts)

    rouge = Rouge()
    try:
        rouge_scores = rouge.get_scores(pred_texts, ref_texts, avg=True, ignore_empty=True)
    except ValueError:
        rouge_scores = "predictions are empty"

    return {
        "corpus_bleu": bleu,
        "sacrebleu": sacrebleu,
        "wer": wer,
        "cer": cer,
        "rouge": rouge_scores,
    }


if __name__ == "__main__":
    args = get_config("eval_decoding")
    with open(args["config_path"], "r", encoding="utf-8") as fp:
        training_config = json.load(fp)

    seed_val = 20
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    device = torch.device(args["cuda"] if torch.cuda.is_available() else "cpu")
    print(f"[INFO] using device: {device}")

    tokenizer = BartTokenizer.from_pretrained(training_config["llm_name"])
    data_root = args["data_root"] if args["data_root"] else training_config.get("data_root", "./dataset/ZuCo")
    datasets = build_dataset(training_config["task_name"], data_root)
    test_set = ZuCo_dataset(
        datasets,
        phase="test",
        tokenizer=tokenizer,
        subject=training_config["subjects"],
        bands=training_config["eeg_bands"],
        setting="unique_sent",
        test_input=args["test_input"],
        max_len=args.get("max_len", training_config.get("max_len", 56)),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args["batch_size"],
        shuffle=False,
        num_workers=args["num_workers"],
    )
    print(f"[INFO] test size: {len(test_set)}")

    temporal_cfg = {
        "hidden_dim": training_config["temporal_hidden_dim"],
        "kernel_sizes": training_config["temporal_kernels"],
        "dropout": training_config["temporal_dropout"],
    }
    graph_cfg = {
        "d_model": training_config["d_model"],
        "num_layers": training_config["graph_layers"],
        "num_heads": training_config["graph_heads"],
        "dropout": training_config["graph_dropout"],
        "readout": training_config["graph_readout"],
    }
    contrastive_cfg = {
        "d_text": training_config["contrastive_d_text"],
        "d_proj": training_config["contrastive_d_proj"],
        "temperature": training_config["contrastive_temperature"],
        "text_model": training_config["text_model"],
    }

    model = GAET(
        temporal_cfg=temporal_cfg,
        graph_cfg=graph_cfg,
        contrastive_cfg=contrastive_cfg,
        llm_name=training_config["llm_name"],
        num_query_tokens=training_config["num_query_tokens"],
        stage="generate",
    )
    state_dict = torch.load(args["checkpoint_path"], map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.set_stage("generate")
    model.to(device)
    model.eval()

    os.makedirs("./results", exist_ok=True)
    os.makedirs("./score_results", exist_ok=True)

    task_name = training_config["task_name"]
    model_tag = "GAET"
    if args["test_input"] == "EEG" and args["train_input"] == "EEG":
        result_path = f"./results/{task_name}-{model_tag}-all_decoding_results.txt"
        score_path = f"./score_results/{task_name}-{model_tag}.txt"
    else:
        result_path = f"./results/{task_name}-{model_tag}-{args['train_input']}_{args['test_input']}-all_decoding_results.txt"
        score_path = f"./score_results/{task_name}-{model_tag}-{args['train_input']}_{args['test_input']}.txt"

    pred_texts: List[str] = []
    ref_texts: List[str] = []
    start = time.time()

    with open(result_path, "w", encoding="utf-8") as fp:
        with torch.no_grad():
            for batch in tqdm(test_loader):
                batch = to_device(batch, device)
                pred_ids = model.generate(
                    eeg_signals=batch["eeg_signals"],
                    adjacency=batch["adjacency"],
                    max_length=args["max_len"],
                    num_beams=args["num_beams"],
                    do_sample=args["do_sample"],
                )
                batch_preds = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
                batch_refs = [tokenizer.decode(t.detach().cpu().tolist(), skip_special_tokens=True) for t in batch["target_ids"]]

                for p, r in zip(batch_preds, batch_refs):
                    pred_texts.append(p)
                    ref_texts.append(r)
                    fp.write(f"target string: {r}\n")
                    fp.write(f"predicted string: {p}\n")
                    fp.write("################################################\n\n")

    metrics = compute_metrics(pred_texts, ref_texts)
    elapsed = (time.time() - start) / 60.0
    print(f"[INFO] eval done in {elapsed:.2f} min")
    print(metrics)

    with open(score_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(metrics, ensure_ascii=False) + "\n")
