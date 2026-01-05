#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Experiment 2 — Imports
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights


# In[2]:


# Device and Reproducibility
Device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", Device)

def SeedEverything(SeedValue: int = 42) -> None:
    """
    I use this during debugging when I want reproducible results.
    If I want randomness, I simply do not call this function.
    """
    random.seed(SeedValue)
    np.random.seed(SeedValue)
    torch.manual_seed(SeedValue)
    torch.cuda.manual_seed_all(SeedValue)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# In[3]:


# Model Construction / Loading
def BuildResNet18(UsePretrainedInit: bool) -> torch.nn.Module:
    """
    I build ResNet18 and replace the final layer for MNIST (10 classes).
    """
    Weights = ResNet18_Weights.IMAGENET1K_V1 if UsePretrainedInit else None
    Model = resnet18(weights=Weights)
    Model.fc = nn.Linear(Model.fc.in_features, 10)
    return Model.to(Device)

def LoadModelFromCheckpoint(CheckpointPath: str, UsePretrainedInit: bool) -> torch.nn.Module:
    """
    Loads a trained model weights from disk and switches to eval mode.
    """
    Model = BuildResNet18(UsePretrainedInit)
    State = torch.load(CheckpointPath, map_location=Device)
    Model.load_state_dict(State)
    Model.eval()
    return Model


# In[4]:


# Dataset + Normalization
DataDir = "./data"

# ResNet18 pretrained weights expect ImageNet normalization
Weights = ResNet18_Weights.IMAGENET1K_V1
Mean, Std = Weights.transforms().mean, Weights.transforms().std

MeanTensor = torch.tensor(Mean).view(3, 1, 1)
StdTensor  = torch.tensor(Std).view(3, 1, 1)

def NormalizeForModel(Image01: torch.Tensor) -> torch.Tensor:
    """
    Image01: [3,H,W] in [0,1]
    Returns ImageNet-normalized tensor (required for ResNet18 pretrained).
    """
    return (Image01 - MeanTensor) / StdTensor

TestTransformNoNorm = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor()   # stays in [0,1]
])

TestDatasetNoNorm = MNIST(root=DataDir, train=False, download=True, transform=TestTransformNoNorm)
print("MNIST Test Size:", len(TestDatasetNoNorm))


# In[5]:


# Load Best Model 
BestModelPath = "./outputs/models/Pretrained_best.pt"
BestModelIsPretrainedInit = True

ModelBest = LoadModelFromCheckpoint(BestModelPath, UsePretrainedInit=BestModelIsPretrainedInit)
print("Loaded model:", BestModelPath)


# In[6]:


#Prediction + Entropy + Visualization
def PredictProbaOneModel(Model: torch.nn.Module, BatchImage01: torch.Tensor) -> torch.Tensor:
    """
    BatchImage01: [B,3,H,W] in [0,1]
    Returns probabilities [B,10]
    """
    BatchNorm = torch.stack([NormalizeForModel(x) for x in BatchImage01], dim=0).to(Device)
    with torch.no_grad():
        Logits = Model(BatchNorm)
        Probs = torch.softmax(Logits, dim=1).cpu()
    return Probs

def PredictBatch(Model: torch.nn.Module, BatchImage01: torch.Tensor) -> torch.Tensor:
    """
    Returns predicted labels [B]
    """
    Probs = PredictProbaOneModel(Model, BatchImage01)
    return Probs.argmax(dim=1)

def EntropyFromProbs(Probs: torch.Tensor, Eps: float = 1e-8) -> torch.Tensor:
    """
    I use predictive entropy to quantify uncertainty.
    Higher entropy => less confident predictions.
    """
    return -torch.sum(Probs * torch.log(Probs + Eps), dim=1)

def PlotGrid(Images01: torch.Tensor, YTrue: torch.Tensor, YPred: torch.Tensor, Title: str, SavePath: str = None) -> None:
    """
    Displays a 4x5 grid of 20 images.
    If SavePath is provided, I save the figure for my report.
    """
    Fig, Axes = plt.subplots(4, 5, figsize=(10, 8))
    Axes = Axes.flatten()

    for i, Ax in enumerate(Axes):
        Img = Images01[i].permute(1, 2, 0).cpu().numpy()
        Ax.imshow(Img)
        Ax.set_title(f"T:{YTrue[i].item()} P:{YPred[i].item()}", fontsize=9)
        Ax.axis("off")

    plt.suptitle(Title)
    plt.tight_layout()

    if SavePath is not None:
        plt.savefig(SavePath, dpi=200, bbox_inches="tight")
        print("Saved:", SavePath)

    plt.show()
    plt.close(Fig)


# In[7]:


# Evaluation (20 random images)
IndexList = torch.randperm(len(TestDatasetNoNorm))[:20].tolist()
Samples = [TestDatasetNoNorm[i] for i in IndexList]

XClean = torch.stack([s[0] for s in Samples], dim=0)   # [20,3,H,W] in [0,1]
YTrue  = torch.tensor([s[1] for s in Samples], dtype=torch.long)

ProbsClean = PredictProbaOneModel(ModelBest, XClean)
PredClean = ProbsClean.argmax(dim=1)

AccClean = (PredClean == YTrue).float().mean().item()
EntClean = EntropyFromProbs(ProbsClean).mean().item()

print(f"Clean Acc on 20 images: {AccClean*100:.2f}%")
print(f"Clean Mean Entropy: {EntClean:.4f}")

PlotGrid(XClean, YTrue, PredClean, Title="Exp2: 20 Random Clean MNIST Test Images")


# In[11]:


# Noise Robustness + Uncertainty (Accuracy + Entropy)
SaveDirExp2 = "Exp2Results"
os.makedirs(SaveDirExp2, exist_ok=True)

NoiseLevels = [0.0, 0.05, 0.10, 0.20, 0.30]

AccByNoise = []
EntByNoise = []

for Lvl in NoiseLevels:
    Noise = torch.randn_like(XClean)
    XNoisy = (XClean + Noise * Lvl).clamp(0, 1)

    ProbsNoisy = PredictProbaOneModel(ModelBest, XNoisy)
    PredNoisy = ProbsNoisy.argmax(dim=1)

    Acc = (PredNoisy == YTrue).float().mean().item()
    Ent = EntropyFromProbs(ProbsNoisy).mean().item()

    AccByNoise.append(Acc)
    EntByNoise.append(Ent)

    GridPath = os.path.join(SaveDirExp2, f"Exp2Noise_{Lvl:.2f}.png")

    PlotGrid(
        XNoisy,
        YTrue,
        PredNoisy,
        Title=f"Exp2: Noise={Lvl:.2f} | Acc={Acc*100:.1f}% | Entropy={Ent:.3f}",
        SavePath=GridPath
    )


# In[10]:


# Save Metrics + Plot Curves
DfExp2 = pd.DataFrame({
    "NoiseLevel": NoiseLevels,
    "Accuracy": AccByNoise,
    "Entropy": EntByNoise
})

CsvPath = os.path.join(SaveDirExp2, "Exp2Metrics.csv")
DfExp2.to_csv(CsvPath, index=False)
print("Saved metrics:", CsvPath)

plt.figure(figsize=(7, 5))
plt.plot(NoiseLevels, AccByNoise, marker="o", label="Accuracy")
plt.plot(NoiseLevels, EntByNoise, marker="o", label="Entropy")
plt.xlabel("White Noise Level")
plt.ylabel("Value")
plt.title("Exp2: Accuracy and Uncertainty vs Noise")
plt.grid(True)
plt.legend()

CurvePath = os.path.join(SaveDirExp2, "Exp2AccEntropyCurve.png")
plt.savefig(CurvePath, dpi=200, bbox_inches="tight")
plt.show()

print("Saved curve plot:", CurvePath)

DfExp2


# In[12]:


# (1) Accuracy vs Noise
# (2) Entropy vs Noise
# (3) Accuracy vs Entropy


# 1) Accuracy vs Noise
plt.figure(figsize=(7, 5))
plt.plot(NoiseLevels, AccByNoise, marker="o")
plt.xlabel("White Noise Level")
plt.ylabel("Accuracy (on 20 images)")
plt.title("Exp2: Accuracy vs Noise Level")
plt.grid(True)

AccNoisePath = os.path.join(SaveDirExp2, "AccVsNoise.png")
plt.savefig(AccNoisePath, dpi=200, bbox_inches="tight")
plt.show()
print("Saved:", AccNoisePath)


# 2) Entropy vs Noise
plt.figure(figsize=(7, 5))
plt.plot(NoiseLevels, EntByNoise, marker="o")
plt.xlabel("White Noise Level")
plt.ylabel("Predictive Entropy")
plt.title("Exp2: Entropy vs Noise Level")
plt.grid(True)

EntNoisePath = os.path.join(SaveDirExp2, "EntVsNoise.png")
plt.savefig(EntNoisePath, dpi=200, bbox_inches="tight")
plt.show()
print("Saved:", EntNoisePath)


# 3) Accuracy vs Entropy
plt.figure(figsize=(7, 5))
plt.plot(EntByNoise, AccByNoise, marker="o")
plt.xlabel("Predictive Entropy")
plt.ylabel("Accuracy (on 20 images)")
plt.title("Exp2: Accuracy vs Entropy")
plt.grid(True)

AccEntPath = os.path.join(SaveDirExp2, "AccVsEntropy.png")
plt.savefig(AccEntPath, dpi=200, bbox_inches="tight")
plt.show()
print("Saved:", AccEntPath)


# In[13]:


plt.figure(figsize=(7, 5))
plt.scatter(EntByNoise, AccByNoise)
plt.xlabel("Predictive Entropy")
plt.ylabel("Accuracy (on 20 images)")
plt.title("Exp2: Accuracy vs Entropy (scatter)")
plt.grid(True)
plt.savefig(AccEntPath, dpi=200, bbox_inches="tight")
plt.show()


# In[ ]:




