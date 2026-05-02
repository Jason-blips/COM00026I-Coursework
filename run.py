import argparse
import csv
import datetime as dt
import os
import re
import subprocess
import sys


# 仅做超参数微调：不改模型结构，不改数据集，不改训练代码逻辑
EXPERIMENTS = [
    {
        "name": "B1_single_stage_lr045",
        "args": [
            "--no-two_stage",
            "--lr", "0.045",
            "--warmup_epochs", "5",
            "--weight_decay", "5e-4",
            "--momentum", "0.9",
            "--patience", "5",
        ],
    },
    {
        "name": "B2_single_stage_lr055",
        "args": [
            "--no-two_stage",
            "--lr", "0.055",
            "--warmup_epochs", "5",
            "--weight_decay", "5e-4",
            "--momentum", "0.9",
            "--patience", "5",
        ],
    },
    {
        "name": "B3_single_stage_warmup3",
        "args": [
            "--no-two_stage",
            "--lr", "0.05",
            "--warmup_epochs", "3",
            "--weight_decay", "5e-4",
            "--momentum", "0.9",
            "--patience", "5",
        ],
    },
    {
        "name": "B4_single_stage_wd3e4",
        "args": [
            "--no-two_stage",
            "--lr", "0.05",
            "--warmup_epochs", "5",
            "--weight_decay", "3e-4",
            "--momentum", "0.9",
            "--patience", "5",
        ],
    },
    {
        "name": "B5_single_stage_wd8e4",
        "args": [
            "--no-two_stage",
            "--lr", "0.05",
            "--warmup_epochs", "5",
            "--weight_decay", "8e-4",
            "--momentum", "0.9",
            "--patience", "5",
        ],
    },
    {
        "name": "B6_two_stage_short_rescue",
        "args": [
            "--two_stage",
            "--stage1_epochs", "2",
            "--stage1_lr", "3e-4",
            "--stage1_weight_decay", "1e-4",
            "--stage1_warmup_epochs", "1",
            "--lr", "0.055",
            "--warmup_epochs", "4",
            "--weight_decay", "5e-4",
            "--momentum", "0.9",
            "--patience", "5",
        ],
    },
]


TEST_RE = re.compile(r"Test Loss:\s*([0-9.]+)\s+Test Acc:\s*([0-9.]+)")
VAL_RE = re.compile(r"Val Loss:\s*([0-9.]+)\s+Val Acc:\s*([0-9.]+)")


def parse_metrics(lines):
    test_loss, test_acc = None, None
    best_val_acc = None
    for line in lines:
        mt = TEST_RE.search(line)
        if mt:
            test_loss = float(mt.group(1))
            test_acc = float(mt.group(2))
        mv = VAL_RE.search(line)
        if mv:
            val_acc = float(mv.group(2))
            if best_val_acc is None or val_acc > best_val_acc:
                best_val_acc = val_acc
    return test_loss, test_acc, best_val_acc


def run_one(exp, cfg):
    exp_dir = os.path.join(cfg["out_dir"], exp["name"])
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    log_path = os.path.join(exp_dir, "train.log")

    cmd = [
        cfg["python"],
        "train.py",
        "--epochs", str(cfg["epochs"]),
        "--data_root", cfg["data_root"],
        "--batch_size", str(cfg["batch_size"]),
        "--num_workers", str(cfg["num_workers"]),
        "--val_ratio", str(cfg["val_ratio"]),
        "--seed", str(cfg["seed"]),
        "--save_dir", ckpt_dir,
    ] + exp["args"]

    print("\n" + "=" * 90)
    print(f"Running {exp['name']}")
    print(" ".join(cmd))
    print("=" * 90)

    lines = []
    with open(log_path, "w", encoding="utf-8") as f:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            f.write(line)
            lines.append(line.rstrip("\n"))
        code = proc.wait()

    test_loss, test_acc, best_val_acc = parse_metrics(lines)
    return {
        "experiment": exp["name"],
        "return_code": code,
        "best_val_acc": best_val_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "log_path": log_path.replace("\\", "/"),
    }


def write_summary(path, results):
    fields = ["experiment", "return_code", "best_val_acc", "test_loss", "test_acc", "log_path"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow(r)


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter-only comparison runner.")
    parser.add_argument("--python", default=sys.executable, help="Python executable path")
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_root", default="results/auto_runs")
    parser.add_argument("--only", nargs="*", default=[], help="Only run selected experiment names")
    args = parser.parse_args()

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.out_root, ts)
    os.makedirs(out_dir, exist_ok=True)

    selected = EXPERIMENTS
    if args.only:
        wanted = set(args.only)
        selected = [x for x in EXPERIMENTS if x["name"] in wanted]
        if not selected:
            print("No experiment matched --only")
            return

    cfg = {
        "python": args.python,
        "data_root": args.data_root,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "out_dir": out_dir,
    }

    results = []
    for exp in selected:
        results.append(run_one(exp, cfg))

    summary_path = os.path.join(out_dir, "summary.csv")
    write_summary(summary_path, results)

    valid = [x for x in results if x["test_acc"] is not None]
    valid.sort(key=lambda x: x["test_acc"], reverse=True)
    print("\nRanking by test_acc:")
    for idx, row in enumerate(valid, 1):
        print(f"{idx}. {row['experiment']} | test_acc={row['test_acc']:.4f} | best_val_acc={row['best_val_acc']}")

    print(f"\nSummary: {summary_path}")


if __name__ == "__main__":
    main()
