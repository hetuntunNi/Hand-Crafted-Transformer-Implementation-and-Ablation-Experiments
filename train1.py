import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from my_transformer import Transformer
import argparse
import requests
import numpy as np
from tqdm import tqdm
import math
import os
import pickle
import matplotlib.pyplot as plt


# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)


# 配置类
class Config:
    def __init__(self, args):
        # 数据参数
        self.seq_length = 256
        self.batch_size = 128
        self.stride = 32

        # 模型参数
        self.d_model = 128
        self.max_len = 256
        self.n_heads = 1
        self.ffn_hidden = 512
        self.n_layers = 2
        self.drop_prob = 0.2

        # 消融实验参数
        self.use_positional_encoding = args.use_positional_encoding
        self.use_residual = not args.no_residual

        # 训练参数
        self.epochs = 20
        self.learning_rate = 0.00005
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.print_every = 100
        self.save_every = 5

        # 数据路径
        self.data_dir = "datasets"
        self.raw_data_file = os.path.join(self.data_dir, "tiny_shakespeare.txt")
        self.train_file = os.path.join(self.data_dir, "train.txt")
        self.val_file = os.path.join(self.data_dir, "val.txt")
        self.vocab_file = os.path.join(self.data_dir, "vocab.pkl")
        self.model_dir = os.path.join(self.data_dir, "models")
        self.results_dir = os.path.join(self.data_dir, "results")

        # 确保目录存在
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)


# 数据集定义
class ShakespeareDataset(Dataset):
    def __init__(self, text, seq_length, vocab):
        self.text = text
        self.seq_length = seq_length
        self.vocab = vocab
        self.data = [self.vocab.get(ch, self.vocab['<unk>']) for ch in text]

    def __len__(self):
        return max(1, len(self.data) - self.seq_length)

    def __getitem__(self, idx):
        seq = self.data[idx: idx + self.seq_length + 1]
        if len(seq) < self.seq_length + 1:
            seq += [self.vocab['<pad>']] * (self.seq_length + 1 - len(seq))
        src = torch.tensor(seq[:-1], dtype=torch.long)
        trg = torch.tensor(seq[1:], dtype=torch.long)
        return src, trg


# 数据加载与词汇表
def ensure_dataset_exists(config):
    if os.path.exists(config.raw_data_file):
        with open(config.raw_data_file, "r", encoding="utf-8") as f:
            text = f.read()
        if not os.path.exists(config.train_file):
            split_and_save_data(text, config)
        return text
    else:
        return download_and_save_dataset(config)


def download_and_save_dataset(config):
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        text = response.text
        with open(config.raw_data_file, "w", encoding="utf-8") as f:
            f.write(text)
        split_and_save_data(text, config)
        return text
    except Exception:
        print("下载失败，使用示例数据")
        return create_sample_dataset(config)


def split_and_save_data(text, config):
    split_idx = int(0.9 * len(text))
    with open(config.train_file, "w", encoding="utf-8") as f:
        f.write(text[:split_idx])
    with open(config.val_file, "w", encoding="utf-8") as f:
        f.write(text[split_idx:])


def create_sample_dataset(config):
    text = """First Citizen: Before we proceed any further, hear me speak.
All: Speak, speak.
First Citizen: You are all resolved rather to die than to famish?
All: Resolved. resolved."""
    with open(config.raw_data_file, "w", encoding="utf-8") as f:
        f.write(text)
    split_and_save_data(text, config)
    return text


def create_or_load_vocab(text, config):
    if os.path.exists(config.vocab_file):
        with open(config.vocab_file, 'rb') as f:
            v = pickle.load(f)
        return v['vocab'], v['idx_to_char'], v['vocab_size']

    chars = sorted(list(set(text)))
    vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
    for i, c in enumerate(chars):
        vocab[c] = i + 4
    idx_to_char = {i: c for c, i in vocab.items()}
    vocab_size = len(vocab)

    with open(config.vocab_file, 'wb') as f:
        pickle.dump({'vocab': vocab, 'idx_to_char': idx_to_char, 'vocab_size': vocab_size}, f)
    return vocab, idx_to_char, vocab_size


# 模型训练
def create_model(vocab_size, config):
    model = Transformer(
        src_pad_idx=0,
        trg_pad_idx=0,
        enc_voc_size=vocab_size,
        dec_voc_size=vocab_size,
        d_model=config.d_model,
        max_len=config.max_len,
        n_heads=config.n_heads,
        ffn_hidden=config.ffn_hidden,
        n_layers=config.n_layers,
        drop_prob=config.drop_prob,
        device=config.device,
        use_positional_encoding=config.use_positional_encoding,
        use_residual=config.use_residual
    )
    return model


def calculate_perplexity(loss):
    try:
        return math.exp(loss)
    except OverflowError:
        return float("inf")


def calculate_accuracy(output, targets, ignore_index=0):
    preds = output.argmax(dim=-1)
    mask = (targets != ignore_index)
    correct = (preds == targets) & mask
    return correct.sum().float() / mask.sum().float()


def train_epoch(model, loader, criterion, optimizer, epoch, vocab_size, config):
    model.train()
    total_loss, total_acc = 0, 0
    progress = tqdm(loader, desc=f"Epoch {epoch + 1}")
    for src, trg in progress:
        src, trg = src.to(config.device), trg.to(config.device)
        optimizer.zero_grad()
        out = model.encoder(src, None)
        if not hasattr(model, "lm_head"):
            model.lm_head = nn.Linear(config.d_model, vocab_size).to(config.device)
        logits = model.lm_head(out)
        loss = criterion(logits.view(-1, vocab_size), trg.view(-1))
        acc = calculate_accuracy(logits.view(-1, vocab_size), trg.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_acc += acc.item()
        progress.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc.item():.2%}")
    avg_loss = total_loss / len(loader)
    avg_acc = total_acc / len(loader)
    return avg_loss, avg_acc, calculate_perplexity(avg_loss)


def evaluate(model, loader, criterion, vocab_size, config):
    model.eval()
    total_loss, total_acc = 0, 0
    with torch.no_grad():
        for src, trg in loader:
            src, trg = src.to(config.device), trg.to(config.device)
            out = model.encoder(src, None)
            logits = model.lm_head(out)
            loss = criterion(logits.view(-1, vocab_size), trg.view(-1))
            acc = calculate_accuracy(logits.view(-1, vocab_size), trg.view(-1))
            total_loss += loss.item()
            total_acc += acc.item()
    avg_loss = total_loss / len(loader)
    avg_acc = total_acc / len(loader)
    return avg_loss, avg_acc, calculate_perplexity(avg_loss)


def save_metrics(train_metrics, val_metrics, config, exp_name):
    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(train_metrics["loss"], label="Train Loss")
    plt.plot(val_metrics["loss"], label="Val Loss")
    plt.legend(); plt.grid(); plt.title("Loss Curve")

    plt.subplot(3, 1, 2)
    plt.plot(train_metrics["accuracy"], label="Train Acc")
    plt.plot(val_metrics["accuracy"], label="Val Acc")
    plt.legend(); plt.grid(); plt.title("Accuracy Curve")

    plt.subplot(3, 1, 3)
    plt.plot(train_metrics["perplexity"], label="Train PPL")
    plt.plot(val_metrics["perplexity"], label="Val PPL")
    plt.legend(); plt.grid(); plt.title("Perplexity Curve")

    plt.tight_layout()
    fig_path = os.path.join(config.results_dir, f"{exp_name}_metrics.png")
    plt.savefig(fig_path)
    print(f"保存指标曲线: {fig_path}")

    data_path = os.path.join(config.results_dir, f"{exp_name}_metrics.pkl")
    with open(data_path, "wb") as f:
        pickle.dump({"train": train_metrics, "val": val_metrics}, f)
    print(f"指标数据已保存: {data_path}")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_positional_encoding", action="store_true", help="启用位置编码")
    parser.add_argument("--no_residual", action="store_true", help="禁用残差连接")
    args = parser.parse_args()

    config = Config(args)
    exp_name = f"pos_0_res_0"
    print(f"\n消融实验配置: {exp_name}\n")

    text = ensure_dataset_exists(config)
    vocab, idx_to_char, vocab_size = create_or_load_vocab(text, config)
    with open(config.train_file, "r", encoding="utf-8") as f:
        train_text = f.read()
    with open(config.val_file, "r", encoding="utf-8") as f:
        val_text = f.read()

    train_loader = DataLoader(ShakespeareDataset(train_text, config.seq_length, vocab), batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(ShakespeareDataset(val_text, config.seq_length, vocab), batch_size=config.batch_size)

    model = create_model(vocab_size, config).to(config.device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    train_metrics = {"loss": [], "accuracy": [], "perplexity": []}
    val_metrics = {"loss": [], "accuracy": [], "perplexity": []}

    best_val_loss = float("inf")

    for epoch in range(config.epochs):
        tr_loss, tr_acc, tr_ppl = train_epoch(model, train_loader, criterion, optimizer, epoch, vocab_size, config)
        val_loss, val_acc, val_ppl = evaluate(model, val_loader, criterion, vocab_size, config)

        train_metrics["loss"].append(tr_loss)
        train_metrics["accuracy"].append(tr_acc)
        train_metrics["perplexity"].append(tr_ppl)

        val_metrics["loss"].append(val_loss)
        val_metrics["accuracy"].append(val_acc)
        val_metrics["perplexity"].append(val_ppl)

        print(f"\nEpoch {epoch + 1}/{config.epochs}:")
        print(f"训练 - Loss {tr_loss:.4f}, Acc {tr_acc:.2%}, PPL {tr_ppl:.2f}")
        print(f"验证 - Loss {val_loss:.4f}, Acc {val_acc:.2%}, PPL {val_ppl:.2f}")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(config.model_dir, f"{exp_name}_best.pth")
            torch.save(model.state_dict(), best_path)
            print(f"保存最佳模型: {best_path}")

    # 训练完成保存最终模型与指标
    final_path = os.path.join(config.model_dir, f"{exp_name}_final.pth")
    torch.save(model.state_dict(), final_path)
    print(f"保存最终模型: {final_path}")

    save_metrics(train_metrics, val_metrics, config, exp_name)


if __name__ == "__main__":
    main()
