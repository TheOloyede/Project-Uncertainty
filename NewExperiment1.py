#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import torchvision
import torchvision.transforms as T
from torchvision.models import resnet18, ResNet18_Weights

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)

DATA_DIR = "./data"           
OUT_DIR  = "./outputs"        
MODEL_DIR = os.path.join(OUT_DIR, "models")
TB_DIR    = os.path.join(OUT_DIR, "tensorboard")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(TB_DIR, exist_ok=True)


# In[2]:


SEED = None

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if SEED is not None:
    seed_everything(SEED)
    print(f"seed fixed to {SEED}")
else:
    print("No seed fixed (non-deterministic run).")


# In[3]:


weights = ResNet18_Weights.IMAGENET1K_V1
mean, std = weights.transforms().mean, weights.transforms().std

train_tf = T.Compose([
    T.Grayscale(num_output_channels=3),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean, std),
])

test_tf = T.Compose([
    T.Grayscale(num_output_channels=3),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean, std),
])

# Safer workers setting (Windows vs Colab/Linux)
num_workers = 0 if os.name == "nt" else 2

train_ds = torchvision.datasets.MNIST(DATA_DIR, train=True, download=True, transform=train_tf)
test_ds  = torchvision.datasets.MNIST(DATA_DIR, train=False, download=True, transform=test_tf)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=num_workers)
test_loader  = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=num_workers)

print("Train size:", len(train_ds), "| Test size:", len(test_ds))


# In[4]:


def build_resnet18(pretrained: bool):
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model.to(DEVICE)


# In[5]:


def accuracy(logits, targets):
    return (logits.argmax(dim=1) == targets).float().mean().item()

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    loss_sum, acc_sum = 0.0, 0.0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss_sum += criterion(out, y).item()
        acc_sum += accuracy(out, y)
    return loss_sum / len(loader), acc_sum / len(loader)

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    loss_sum, acc_sum = 0.0, 0.0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
        acc_sum += accuracy(out, y)
    return loss_sum / len(loader), acc_sum / len(loader)


# In[6]:


def train_model(run_name, pretrained, epochs, lr, log_iter_every=0):
    model = build_resnet18(pretrained)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    writer = SummaryWriter(os.path.join(TB_DIR, run_name))
    best_acc = 0.0
    best_path = os.path.join(MODEL_DIR, f"{run_name}_best.pt")

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    global_step = 0

    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum, acc_sum = 0.0, 0.0

        # ---- train one epoch (with optional iteration logging) ----
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            batch_acc = accuracy(out, y)
            loss_sum += loss.item()
            acc_sum += batch_acc

            if log_iter_every and (batch_idx % log_iter_every == 0):
                writer.add_scalar("train_iter/loss", loss.item(), global_step)
                writer.add_scalar("train_iter/acc", batch_acc, global_step)

            global_step += 1

        tr_loss = loss_sum / len(train_loader)
        tr_acc  = acc_sum / len(train_loader)

        te_loss, te_acc = evaluate(model, test_loader, criterion)

        # ---- epoch-level logging ----
        writer.add_scalar("train/loss", tr_loss, epoch)
        writer.add_scalar("train/acc", tr_acc, epoch)
        writer.add_scalar("val/loss", te_loss, epoch)
        writer.add_scalar("val/acc", te_acc, epoch)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(te_loss)
        history["val_acc"].append(te_acc)

        if te_acc > best_acc:
            best_acc = te_acc
            torch.save(model.state_dict(), best_path)

        print(
            f"{run_name} | Epoch {epoch}: "
            f"train_loss={tr_loss:.4f} train_acc={tr_acc*100:.2f}% | "
            f"val_loss={te_loss:.4f} val_acc={te_acc*100:.2f}%"
        )

    writer.close()
    print(f"Best accuracy for {run_name}: {best_acc*100:.2f}%")
    print("Saved:", best_path)

    return best_path, history


# In[ ]:


exp1_scratch_path, hist_scratch = train_model(
    "Without_Pretraining",
    pretrained=False,
    epochs=10,
    lr=1e-3,
    log_iter_every= 50  # set e.g. 50 if you want iteration logging
)

exp1_pretrained_path, hist_pretrained = train_model(
    "Pretrained",
    pretrained=True,
    epochs=8,
    lr=1e-4,
    log_iter_every= 50  # set e.g. 50 if you want iteration logging
)


# In[ ]:


import matplotlib.pyplot as plt

# --- Plot: Loss vs Epoch (Scratch vs Pretrained) ---
plt.figure()
plt.plot(hist_scratch["train_loss"], label="Scratch - Train loss")
plt.plot(hist_scratch["val_loss"],   label="Scratch - Val loss")
plt.plot(hist_pretrained["train_loss"], label="Pretrained - Train loss")
plt.plot(hist_pretrained["val_loss"],   label="Pretrained - Val loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("ResNet18 on MNIST: Loss vs Epoch (Scratch vs Pretrained)")
plt.legend()
plt.show()


# In[ ]:


plot_path = os.path.join(OUT_DIR, "loss_curve_scratch_vs_pretrained.png")
plt.savefig(plot_path, dpi=200, bbox_inches="tight")
print("Saved plot to:", plot_path)


# In[ ]:




