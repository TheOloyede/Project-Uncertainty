#!/usr/bin/env python
# coding: utf-8

# In[19]:


import os
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights


# In[20]:


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)


# In[21]:


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
    print(f"Seed fixed to {SEED}")
else:
    print("No seed fixed (non-deterministic run).")


# In[22]:


def build_resnet18(pretrained_init: bool):
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained_init else None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model.to(DEVICE)

def load_model_from_checkpoint(ckpt_path: str, pretrained_init: bool):
    model = build_resnet18(pretrained_init)
    state = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


# In[23]:


DATA_DIR = "./data"  

weights = ResNet18_Weights.IMAGENET1K_V1
mean, std = weights.transforms().mean, weights.transforms().std

# Testing dataset WITHOUT normalization 
test_transform_no_norm = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor()  
])

test_dataset_no_norm = MNIST(root=DATA_DIR, train=False, download=True, transform=test_transform_no_norm)
print("Test size:", len(test_dataset_no_norm))

mean_t = torch.tensor(mean).view(3, 1, 1)
std_t  = torch.tensor(std).view(3, 1, 1)

def normalize_for_model(x01: torch.Tensor) -> torch.Tensor:
    """x01 in [0,1], shape [3,H,W] -> normalized tensor for resnet"""
    return (x01 - mean_t) / std_t


# In[24]:


BEST_MODEL_PATH = "./outputs/models/Pretrained_best.pt"  
BEST_MODEL_PRETRAINED_INIT = True  

model_best = load_model_from_checkpoint(BEST_MODEL_PATH, pretrained_init=BEST_MODEL_PRETRAINED_INIT)
print("Loaded model:", BEST_MODEL_PATH)


# In[25]:


def predict_batch(model, x01_batch: torch.Tensor) -> torch.Tensor:
    """
    x01_batch: [B,3,H,W] in [0,1]
    returns predicted labels [B]
    """
    xb = torch.stack([normalize_for_model(x) for x in x01_batch], dim=0).to(DEVICE)
    with torch.no_grad():
        logits = model(xb)
        pred = logits.argmax(dim=1).cpu()
    return pred

# Pick 20 random test images
idx = torch.randperm(len(test_dataset_no_norm))[:20].tolist()
samples = [test_dataset_no_norm[i] for i in idx]

x_clean = torch.stack([s[0] for s in samples], dim=0)  # [20,3,H,W], [0,1]
y_true  = torch.tensor([s[1] for s in samples], dtype=torch.long)

pred_clean = predict_batch(model_best, x_clean)
acc_clean = (pred_clean == y_true).float().mean().item()
print("Clean accuracy on these 20:", acc_clean)


# In[26]:


def plot_grid(images01: torch.Tensor, y_true: torch.Tensor, y_pred: torch.Tensor, title: str):
    """
    Show 20 images in a 4x5 grid with true/pred labels.
    images01: [20,3,H,W] in [0,1]
    """
    fig, axes = plt.subplots(4, 5, figsize=(12, 10))
    fig.suptitle(title, fontsize=14)
    axes = axes.flatten()

    for i in range(20):
        img = images01[i].permute(1, 2, 0).clamp(0, 1)  # HWC
        axes[i].imshow(img)
        axes[i].axis("off")
        axes[i].set_title(f"T:{y_true[i].item()} P:{y_pred[i].item()}", fontsize=10)

    plt.tight_layout()
    plt.show()


# In[27]:


plot_grid(
    x_clean,
    y_true,
    pred_clean,
    title=f"Exp2: Clean images (20 random) | Acc={acc_clean*100:.1f}%"
)


# In[28]:


noise_levels = [0.0, 0.05, 0.10, 0.20, 0.30]
acc_by_level = []

for lvl in noise_levels:
    noise = torch.randn_like(x_clean)  # white noise N(0,1)
    x_noisy = (x_clean + noise * lvl).clamp(0, 1)  # image <- image + noise*level

    pred_noisy = predict_batch(model_best, x_noisy)
    acc = (pred_noisy == y_true).float().mean().item()
    acc_by_level.append(acc)

    plot_grid(
        x_noisy,
        y_true,
        pred_noisy,
        title=f"Exp2: White noise level={lvl:.2f} | Acc={acc*100:.1f}%"
    )


# In[29]:


plt.figure()
plt.plot(noise_levels, acc_by_level, marker="o")
plt.xlabel("White noise level")
plt.ylabel("Accuracy on 20 images")
plt.title("Exp2: Accuracy drop vs noise level (20 random MNIST test images)")
plt.ylim(0, 1.05)
plt.grid(True)
plt.show()


# ### **Observation on (Experiment 2):**
# When white noise is added, classification accuracy drops as noise level increases. 

# #### **Is a noticeable visual degradation (to the human eye) necessary to see a significant drop in classification results?**
# NO, A significant accuracy drop occurs at low noise levels and these noise levels do not cause strong visual degradation. Therefore, Humans can still recognize digits while the model already fails

# *From the accuracy versus noise curve, we observe that classification performance degrades rapidly as white noise intensity increases. Importantly, a substantial drop in accuracy is already visible at low noise levels (e.g. 0.05–0.10), even though the digits remain clearly recognizable to the human eye. This indicates that noticeable visual degradation is not required to observe a significant loss in classification performance. The neural network is therefore more sensitive to small perturbations than human perception, highlighting a lack of robustness to additive white noise.*
# 

# 

# In[ ]:




