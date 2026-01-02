#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.datasets import MNIST, KMNIST
import torchvision.transforms as transforms


# In[3]:


import certifi, os, ssl
os.environ["SSL_CERT_FILE"] = certifi.where()


# In[4]:


DATA_DIR = "./data" 

test_transform_no_norm = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # [0,1]
])

mnist_test = MNIST(root=DATA_DIR, train=False, download=True, transform=test_transform_no_norm)
kmnist_test = KMNIST(root=DATA_DIR, train=False, download=True, transform=test_transform_no_norm)

print("MNIST test:", len(mnist_test))
print("KMNIST test:", len(kmnist_test))


# In[5]:


N_SHOW = 20

mn_idx = torch.randperm(len(mnist_test))[:N_SHOW].tolist()
km_idx = torch.randperm(len(kmnist_test))[:N_SHOW].tolist()

mn_samples = [mnist_test[i] for i in mn_idx]
km_samples = [kmnist_test[i] for i in km_idx]

x_mn = torch.stack([s[0] for s in mn_samples], dim=0)   # [20,3,224,224]
y_mn = torch.tensor([s[1] for s in mn_samples], dtype=torch.long)

x_km = torch.stack([s[0] for s in km_samples], dim=0)
y_km = torch.tensor([s[1] for s in km_samples], dtype=torch.long)

print("MNIST batch:", x_mn.shape, y_mn.shape)
print("KMNIST batch:", x_km.shape, y_km.shape)


# In[10]:


# Model paths 
model_paths = [
    "/content/outputs/models/exp3_ens_model1_seed1_best.pt",
    "/content/outputs/models/exp3_ens_model2_seed2_best.pt",
    "/content/outputs/models/exp3_ens_model3_seed3_best.pt",
    "/content/outputs/models/exp3_ens_model4_seed4_best.pt",
    "/content/outputs/models/exp3_ens_model5_seed5_best.pt",
    "/content/outputs/models/exp3_ens_model6_seed6_best.pt",
    "/content/outputs/models/exp3_ens_model7_seed7_best.pt",
]

for p in model_paths:
    print(p, "| exists:", os.path.exists(p))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

# Build + load models

PRETRAINED_INIT = True  

def build_resnet18(pretrained_init: bool):
    model = resnet18(
        weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained_init else None
    )
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model.to(DEVICE)

def load_model_from_checkpoint(path: str, pretrained_init: bool):
    model = build_resnet18(pretrained_init)
    state = torch.load(path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model

models_ens = [load_model_from_checkpoint(p, PRETRAINED_INIT) for p in model_paths]

print("Loaded models:", len(models_ens))


# In[11]:


DATA_DIR = "./data"

test_transform_no_norm = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # [0,1]
])

mnist_test = MNIST(root=DATA_DIR, train=False, download=True, transform=test_transform_no_norm)
kmnist_test = KMNIST(root=DATA_DIR, train=False, download=True, transform=test_transform_no_norm)

print("MNIST test:", len(mnist_test))
print("KMNIST test:", len(kmnist_test))


# In[12]:


N_SHOW = 20

mn_idx = torch.randperm(len(mnist_test))[:N_SHOW].tolist()
km_idx = torch.randperm(len(kmnist_test))[:N_SHOW].tolist()

mn_samples = [mnist_test[i] for i in mn_idx]
km_samples = [kmnist_test[i] for i in km_idx]

x_mn = torch.stack([s[0] for s in mn_samples], dim=0)
y_mn = torch.tensor([s[1] for s in mn_samples], dtype=torch.long)

x_km = torch.stack([s[0] for s in km_samples], dim=0)
y_km = torch.tensor([s[1] for s in km_samples], dtype=torch.long)

print("MNIST batch:", x_mn.shape, y_mn.shape)
print("KMNIST batch:", x_km.shape, y_km.shape)


# In[13]:


# pick best single model 
best_single_idx = 0
for i, p in enumerate(model_paths):
    if "model6" in os.path.basename(p):
        best_single_idx = i
        break

best_model = models_ens[best_single_idx]
print("Best single model:", os.path.basename(model_paths[best_single_idx]))

def predict_proba_one_model(model, x01_batch: torch.Tensor) -> torch.Tensor:
    xb = torch.stack([normalize_for_model(x) for x in x01_batch], dim=0).to(DEVICE)
    with torch.no_grad():
        logits = model(xb)
        probs = F.softmax(logits, dim=1)
    return probs.cpu()  # [B,10]

def entropy_from_probs(probs: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    p = probs.clamp(min=eps)
    return -(p * p.log()).sum(dim=1)  # [B]

def ensemble_probs_probavg(models, x01_batch: torch.Tensor) -> torch.Tensor:
    probs_list = [predict_proba_one_model(m, x01_batch) for m in models]  
    probs_stack = torch.stack(probs_list, dim=0)                          
    return probs_stack.mean(dim=0)                                        


# In[15]:


# ImageNet normalization 
weights = ResNet18_Weights.IMAGENET1K_V1
mean, std = weights.transforms().mean, weights.transforms().std

mean_t = torch.tensor(mean).view(3, 1, 1)
std_t  = torch.tensor(std).view(3, 1, 1)

def normalize_for_model(x01: torch.Tensor) -> torch.Tensor:
    """
    x01: Tensor [3,H,W] in [0,1]
    returns normalized tensor for ResNet18
    """
    return (x01 - mean_t) / std_t

print("normalize_for_model defined.")


# In[16]:


# MNIST
probs_single_mn = predict_proba_one_model(best_model, x_mn)
pred_single_mn = probs_single_mn.argmax(dim=1)
acc_single_mn = (pred_single_mn == y_mn).float().mean().item()
ent_single_mn = entropy_from_probs(probs_single_mn).mean().item()

probs_ens_mn = ensemble_probs_probavg(models_ens, x_mn)
pred_ens_mn = probs_ens_mn.argmax(dim=1)
acc_ens_mn = (pred_ens_mn == y_mn).float().mean().item()
ent_ens_mn = entropy_from_probs(probs_ens_mn).mean().item()

# KMNIST
probs_single_km = predict_proba_one_model(best_model, x_km)
pred_single_km = probs_single_km.argmax(dim=1)
acc_single_km = (pred_single_km == y_km).float().mean().item()
ent_single_km = entropy_from_probs(probs_single_km).mean().item()

probs_ens_km = ensemble_probs_probavg(models_ens, x_km)
pred_ens_km = probs_ens_km.argmax(dim=1)
acc_ens_km = (pred_ens_km == y_km).float().mean().item()
ent_ens_km = entropy_from_probs(probs_ens_km).mean().item()

print("MNIST  | Single acc:", acc_single_mn, "| Ens acc:", acc_ens_mn, "| Single ent:", ent_single_mn, "| Ens ent:", ent_ens_mn)
print("KMNIST | Single acc:", acc_single_km, "| Ens acc:", acc_ens_km, "| Single ent:", ent_single_km, "| Ens ent:", ent_ens_km)


# **What would your conclusions be if you only used a single model?**
# 
# Based on the result,If only a single MNIST-trained model were used:
# * the model generalizes extremely poorly to KMNIST, the accuracy drops from 100% → 5%
# * Predictions on KMNIST are largely incorrect
# However, the model still produces moderate confidence (entropy ≈ 0.59), despite being wrong
# This shows a limitation that a single model can be confidently wrong on out-of-distribution data.

# In[17]:


def plot_grid(images01: torch.Tensor, y_true: torch.Tensor, y_pred: torch.Tensor, title: str):
    fig, axes = plt.subplots(4, 5, figsize=(12, 10))
    fig.suptitle(title, fontsize=14)
    axes = axes.flatten()

    for i in range(20):
        img = images01[i].permute(1, 2, 0).clamp(0, 1)
        axes[i].imshow(img)
        axes[i].axis("off")
        axes[i].set_title(f"T:{y_true[i].item()} P:{y_pred[i].item()}", fontsize=10)

    plt.tight_layout()
    plt.show()

plot_grid(x_mn, y_mn, pred_single_mn, title=f"Exp4: MNIST — Best single | Acc={acc_single_mn*100:.1f}% | Ent={ent_single_mn:.2f}")
plot_grid(x_mn, y_mn, pred_ens_mn,    title=f"Exp4: MNIST — Ensemble prob-avg | Acc={acc_ens_mn*100:.1f}% | Ent={ent_ens_mn:.2f}")

plot_grid(x_km, y_km, pred_single_km, title=f"Exp4: KMNIST — Best single | Acc={acc_single_km*100:.1f}% | Ent={ent_single_km:.2f}")
plot_grid(x_km, y_km, pred_ens_km,    title=f"Exp4: KMNIST — Ensemble prob-avg | Acc={acc_ens_km*100:.1f}% | Ent={ent_ens_km:.2f}")


# ### Out-of-Distribution Generalization:
# 
# On MNIST test images, both the best single model and the ensemble achieve perfect accuracy with very low predictive entropy, indicating confident and correct in-distribution classification. In contrast, performance degrades drastically on KMNIST images, which represent a different character distribution despite similar image format.
# 
# If only a single model is used, one would conclude that the classifier generalizes poorly to KMNIST, while still producing moderately confident but incorrect predictions. This highlights the risk of relying on accuracy or confidence from a single model under distribution shift.
# ### What is the benefit of using multiple models?
# Using multiple independently trained models provides additional reliability through increased predictive uncertainty. On KMNIST, the ensemble exhibits significantly higher entropy, reflecting strong disagreement between models and offering a clear indication of out-of-distribution inputs. Although accuracy remains low, the ensemble enables uncertainty-aware decision making, demonstrating the benefit of multiple models for robustness and out-of-distribution awareness rather than raw performance.
