"""
Train FMF data with demo encoders (ResNet18 + BERT) and FC decision heads (no fuzzy TSK).
Training/eval flow mirrors tools/train_demo_tmc.py.
"""
import argparse
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from FMF.datasets import build_dataset
from FMF.utils.metrics import ClassificationMetric
from FMF.utils.parser import load_config


def parse_all_args():
    parser = argparse.ArgumentParser(
        description="Train FMF with demo encoders + FC heads (no fuzzy TSK)."
    )
    parser.add_argument("--cfg", dest="cfg_file", default="configs/CfgForViCu_Cls.yaml", type=str)
    parser.add_argument("--demo-dir", default="demo", type=str)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--lr-factor", default=0.3, type=float)
    parser.add_argument("--lr-patience", default=10, type=int)
    parser.add_argument("--patience", default=20, type=int)
    parser.add_argument("--image-size", default=224, type=int)
    parser.add_argument("--annealing-epoch", default=10, type=int)
    parser.add_argument("--sim-p", default=0.05, type=float)
    parser.add_argument("--gradient-accumulation-steps", default=3, type=int)
    parser.add_argument("--fc-hidden", default=256, type=int, help="Hidden dim of FC decision heads")
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--bert-model", default="demo/bert-base-uncased", type=str)
    parser.add_argument("--max-seq-len", default=128, type=int)
    parser.add_argument("--current-precision", default=2, type=int)
    parser.add_argument("--save-path", default="./state_dict_demo_fc_fmf_best.pth", type=str)
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    return parser.parse_args()


def import_demo_modules(demo_dir):
    if demo_dir not in sys.path:
        sys.path.insert(0, demo_dir)
    try:
        from models_fc import TMC_FC
        from models import ce_loss, sim_loss
    except Exception as exc:
        raise ImportError(
            f"Failed to import TMC_FC from {demo_dir}/models_fc.py. "
            "Ensure demo/models_fc.py exists."
        ) from exc
    return TMC_FC, ce_loss, sim_loss


def resolve_path(path_value):
    if os.path.isabs(path_value):
        return path_value
    return os.path.abspath(os.path.join(PROJECT_ROOT, path_value))


def load_bert_tokenizer(bert_model_path_or_name):
    load_errors = []
    try:
        from pytorch_pretrained_bert import BertTokenizer as PPBBertTokenizer
        tok = PPBBertTokenizer.from_pretrained(bert_model_path_or_name, do_lower_case=True)
        if tok is not None:
            return tok
        load_errors.append("pytorch_pretrained_bert returned None tokenizer")
    except Exception as exc:
        load_errors.append(f"pytorch_pretrained_bert failed: {exc}")

    try:
        from transformers import BertTokenizer as HFBertTokenizer
        tok = HFBertTokenizer.from_pretrained(bert_model_path_or_name, do_lower_case=True)
        if tok is not None:
            return tok
        load_errors.append("transformers returned None tokenizer")
    except Exception as exc:
        load_errors.append(f"transformers failed: {exc}")

    raise RuntimeError(
        "Failed to load BERT tokenizer. "
        f"Given --bert-model: {bert_model_path_or_name}. Details: {' | '.join(load_errors)}"
    )


def build_demo_args(num_classes, user_args):
    return SimpleNamespace(
        model="bert",
        img_embed_pool_type="avg",
        num_image_embeds=1,
        img_hidden_sz=512,
        hidden_sz_bert=768,
        bert_model=user_args.bert_model,
        n_classes=num_classes,
        annealing_epoch=user_args.annealing_epoch,
        fc_hidden=user_args.fc_hidden,
        dropout=user_args.dropout,
    )


class FMFBertAdapterDataset(Dataset):
    def __init__(self, base_dataset, tokenizer, max_seq_len, image_size, current_precision):
        self.base_dataset = base_dataset
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.image_size = image_size
        self.current_precision = current_precision
        self.mean = torch.tensor([0.46777044, 0.44531429, 0.40661017], dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor([0.12221994, 0.12145835, 0.14380469], dtype=torch.float32).view(3, 1, 1)

    def __len__(self):
        return len(self.base_dataset)

    def _current_to_text(self, current):
        if isinstance(current, torch.Tensor):
            current = current.cpu().numpy()
        elif not isinstance(current, np.ndarray):
            current = np.asarray(current)
        steps = []
        for row in current:
            values = [f"{float(v):.{self.current_precision}f}" for v in row]
            steps.append(",".join(values))
        return " [SEP] ".join(steps)

    def _build_bert_inputs(self, text):
        tokens = ["[CLS]"] + self.tokenizer.tokenize(text)
        tokens = tokens[: self.max_seq_len]
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        text_tensor = torch.tensor(token_ids, dtype=torch.long)
        segment_tensor = torch.zeros(len(token_ids), dtype=torch.long)
        mask_tensor = torch.ones(len(token_ids), dtype=torch.long)
        return text_tensor, segment_tensor, mask_tensor

    def _video_to_image(self, video):
        if isinstance(video, np.ndarray):
            video = torch.from_numpy(video)
        elif not isinstance(video, torch.Tensor):
            video = torch.as_tensor(video)
        img = video[:, -1, :, :].float()
        img = F.interpolate(
            img.unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        img = img / 255.0 if img.max() > 1.0 else img
        img = (img - self.mean) / self.std
        return img

    def __getitem__(self, idx):
        video, current, label = self.base_dataset[idx]
        text = self._current_to_text(current)
        txt, segment, mask = self._build_bert_inputs(text)
        img = self._video_to_image(video)
        tgt = torch.tensor(label, dtype=torch.long)
        idx_t = torch.tensor(idx, dtype=torch.long)
        return txt, segment, mask, img, tgt, idx_t


def bert_collate_fn(batch):
    lens = [len(row[0]) for row in batch]
    bsz = len(batch)
    max_len = max(lens)
    text_tensor = torch.zeros(bsz, max_len, dtype=torch.long)
    segment_tensor = torch.zeros(bsz, max_len, dtype=torch.long)
    mask_tensor = torch.zeros(bsz, max_len, dtype=torch.long)
    img_tensor = torch.stack([row[3] for row in batch])
    tgt_tensor = torch.stack([row[4] for row in batch]).long()
    idx_tensor = torch.stack([row[5] for row in batch]).long()
    for i, (txt, seg, msk, _, _, _) in enumerate(batch):
        ln = txt.size(0)
        text_tensor[i, :ln] = txt
        segment_tensor[i, :ln] = seg
        mask_tensor[i, :ln] = msk
    return text_tensor, segment_tensor, mask_tensor, img_tensor, tgt_tensor, idx_tensor


def model_forward(i_epoch, model, demo_args, ce_loss, sim_loss, sim_weight, batch, device):
    txt, segment, mask, img, tgt, idx = batch
    txt = txt.to(device, non_blocking=True)
    segment = segment.to(device, non_blocking=True)
    mask = mask.to(device, non_blocking=True)
    img = img.to(device, non_blocking=True)
    tgt = tgt.to(device, non_blocking=True)
    idx = idx.to(device, non_blocking=True)

    model_batch = (txt, segment, mask, img, tgt, idx)
    bert_alpha, rgb_alpha, fused_alpha, bert_out, rgb_out = model(model_batch, device)
    loss = (
        ce_loss(tgt, bert_alpha, demo_args.n_classes, i_epoch, demo_args.annealing_epoch)
        + ce_loss(tgt, rgb_alpha, demo_args.n_classes, i_epoch, demo_args.annealing_epoch)
        + ce_loss(tgt, fused_alpha, demo_args.n_classes, i_epoch, demo_args.annealing_epoch)
        + sim_loss(bert_out, rgb_out, sim_weight)
    )
    return loss, bert_alpha, rgb_alpha, fused_alpha, tgt


def evaluate(i_epoch, loader, model, demo_args, ce_loss, sim_loss, sim_weight, device):
    model.eval()
    losses, bert_preds, rgb_preds, fused_preds, tgts = [], [], [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, ncols=100):
            loss, bert_alpha, rgb_alpha, fused_alpha, tgt = model_forward(
                i_epoch, model, demo_args, ce_loss, sim_loss, sim_weight, batch, device
            )
            losses.append(loss.item())
            bert_preds.extend(bert_alpha.argmax(dim=1).cpu().tolist())
            rgb_preds.extend(rgb_alpha.argmax(dim=1).cpu().tolist())
            fused_preds.extend(fused_alpha.argmax(dim=1).cpu().tolist())
            tgts.extend(tgt.cpu().tolist())

    cls_metric = ClassificationMetric(numClass=2)
    cls_metric.addBatch(np.array(fused_preds), np.array(tgts))

    return {
        "loss": sum(losses) / max(len(losses), 1),
        "bert_acc": accuracy_score(tgts, bert_preds),
        "rgb_acc": accuracy_score(tgts, rgb_preds),
        "fused_acc": accuracy_score(tgts, fused_preds),
        "acc": cls_metric.Accuracy(),
        "f1": cls_metric.F1Score(),
        "fdr": cls_metric.FalsePositiveRate(),
        "mdr": cls_metric.FalseNegativeRate(),
    }


def main():
    args = parse_all_args()
    args.cfg_file = resolve_path(args.cfg_file)
    args.demo_dir = resolve_path(args.demo_dir)
    args.save_path = resolve_path(args.save_path)
    if not os.path.isabs(args.bert_model):
        candidate = resolve_path(args.bert_model)
        if os.path.isdir(candidate):
            args.bert_model = candidate

    cfg = load_config(args)
    cfg.DATA.PATH_TO_DATA_DIR = resolve_path(cfg.DATA.PATH_TO_DATA_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    TMC_FC, ce_loss, sim_loss = import_demo_modules(args.demo_dir)
    demo_args = build_demo_args(cfg.MODEL.NUM_CLASSES, args)
    tokenizer = load_bert_tokenizer(args.bert_model)

    train_base = build_dataset(name=cfg.TRAIN.DATASET, cfg=cfg, split="train")
    test_base = build_dataset(name=cfg.TEST.DATASET, cfg=cfg, split="test")
    train_set = FMFBertAdapterDataset(
        train_base, tokenizer, args.max_seq_len, args.image_size, args.current_precision
    )
    test_set = FMFBertAdapterDataset(
        test_base, tokenizer, args.max_seq_len, args.image_size, args.current_precision
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=bert_collate_fn,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=bert_collate_fn,
    )

    model = TMC_FC(demo_args).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=args.lr_factor, patience=args.lr_patience, verbose=True
    )
    sim_weight = torch.FloatTensor([args.sim_p / 2.0]).to(device)

    best_acc = -1.0
    n_no_improve = 0
    os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)

    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        train_losses = []
        optimizer.zero_grad()

        for batch in tqdm(train_loader, total=len(train_loader), ncols=100):
            loss, _, _, _, _ = model_forward(
                epoch, model, demo_args, ce_loss, sim_loss, sim_weight, batch, device
            )
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            train_losses.append(loss.item())
            loss.backward()
            global_step += 1
            if global_step % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        metrics = evaluate(epoch, test_loader, model, demo_args, ce_loss, sim_loss, sim_weight, device)
        scheduler.step(metrics["fused_acc"])
        print(
            f"[Epoch {epoch + 1}/{args.epochs}] "
            f"train_loss={sum(train_losses)/max(len(train_losses),1):.4f} "
            f"test_loss={metrics['loss']:.4f} "
            f"bert_acc={metrics['bert_acc']:.4f} "
            f"rgb_acc={metrics['rgb_acc']:.4f} "
            f"fused_acc={metrics['fused_acc']:.4f} "
            f"ACC={metrics['acc']:.4f} "
            f"F1={metrics['f1']:.4f} "
            f"FDR={metrics['fdr']:.4f} "
            f"MDR={metrics['mdr']:.4f}"
        )

        if metrics["fused_acc"] > best_acc:
            best_acc = metrics["fused_acc"]
            n_no_improve = 0
            torch.save(model.state_dict(), args.save_path)
            print(f"[INFO] Saved best model to {args.save_path} (fused_acc={best_acc:.4f})")
        else:
            n_no_improve += 1
            if n_no_improve >= args.patience:
                print("[INFO] Early stop triggered.")
                break

    print(f"[INFO] Done. Best fused_acc={best_acc:.4f}")


if __name__ == "__main__":
    main()
