import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def get_config(case):
    if case == "train_decoding":
        parser = argparse.ArgumentParser(description="GAET training config")

        parser.add_argument("-t", "--task_name", default="task1", required=True)
        parser.add_argument("-s", "--save_path", default="./checkpoints/decoding", required=True)
        parser.add_argument("--data_root", default="./dataset/ZuCo")
        parser.add_argument("-subj", "--subjects", default="ALL")
        parser.add_argument("-band", "--eeg_bands", nargs="+", default=["_t1", "_t2", "_a1", "_a2", "_b1", "_b2", "_g1", "_g2"])
        parser.add_argument("-cuda", "--cuda", default="cuda:0")
        parser.add_argument("-train_input", "--train_input", default="EEG", required=True)

        parser.add_argument("-b", "--batch_size", type=int, default=16)
        parser.add_argument("--max_len", type=int, default=56)
        parser.add_argument("--num_workers", type=int, default=4)
        parser.add_argument("--gradient_clip", type=float, default=1.0)

        parser.add_argument("--stage1_epochs", type=int, default=20)
        parser.add_argument("--stage2_epochs", type=int, default=30)
        parser.add_argument("--stage1_lr", type=float, default=1e-4)
        parser.add_argument("--stage2_lr", type=float, default=2e-5)
        parser.add_argument("--weight_decay", type=float, default=1e-4)

        parser.add_argument("--d_model", type=int, default=256)
        parser.add_argument("--graph_layers", type=int, default=4)
        parser.add_argument("--graph_heads", type=int, default=8)
        parser.add_argument("--graph_dropout", type=float, default=0.1)
        parser.add_argument("--graph_readout", type=str, default="mean", choices=["mean", "max", "meanmax"])

        parser.add_argument("--temporal_hidden_dim", type=int, default=64)
        parser.add_argument("--temporal_kernels", nargs="+", type=int, default=[3, 3, 3])
        parser.add_argument("--temporal_dropout", type=float, default=0.1)

        parser.add_argument("--text_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
        parser.add_argument("--contrastive_d_text", type=int, default=384)
        parser.add_argument("--contrastive_d_proj", type=int, default=256)
        parser.add_argument("--contrastive_temperature", type=float, default=0.07)
        parser.add_argument("--contrastive_weight", type=float, default=0.0)
        parser.add_argument("--contrastive_warmup_epochs", type=int, default=0)

        parser.add_argument("--llm_name", type=str, default="facebook/bart-large")
        parser.add_argument("--num_query_tokens", type=int, default=32)

        args = vars(parser.parse_args())
        return args

    if case == "eval_decoding":
        parser = argparse.ArgumentParser(description="GAET evaluation config")
        parser.add_argument("-checkpoint", "--checkpoint_path", required=True)
        parser.add_argument("-conf", "--config_path", required=True)
        parser.add_argument("--data_root", default="")
        parser.add_argument("-test_input", "--test_input", required=True)
        parser.add_argument("-train_input", "--train_input", required=True)
        parser.add_argument("-cuda", "--cuda", default="cuda:0")
        parser.add_argument("--num_beams", type=int, default=4)
        parser.add_argument("--do_sample", type=str2bool, default=False)
        parser.add_argument("--max_len", type=int, default=56)
        parser.add_argument("--batch_size", type=int, default=1)
        parser.add_argument("--num_workers", type=int, default=4)
        args = vars(parser.parse_args())
        return args

    raise ValueError(f"Unsupported config case: {case}")
