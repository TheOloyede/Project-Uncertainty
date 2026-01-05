#!/usr/bin/env python
# coding: utf-8

# In[8]:


# Experiment 5 — Imports
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
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

Weights = ResNet18_Weights.IMAGENET1K_V1
Mean, Std = Weights.transforms().mean, Weights.transforms().std

MeanTensor = torch.tensor(Mean).view(3, 1, 1)
StdTensor  = torch.tensor(Std).view(3, 1, 1)

def NormalizeForModel(Image01: torch.Tensor) -> torch.Tensor:
    """
    Image01: [3,H,W] in [0,1]
    Returns ImageNet-normalized tensor.
    """
    return (Image01 - MeanTensor) / StdTensor

TestTransformNoNorm = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

TestDatasetNoNorm = MNIST(root=DataDir, train=False, download=True, transform=TestTransformNoNorm)
print("MNIST Test Size:", len(TestDatasetNoNorm))


# In[5]:


# Load 7 models (Ensemble)
ModelPaths = [
    "./outputs/models/Model1Seed1.pt",
    "./outputs/models/Model2Seed2.pt",
    "./outputs/models/Model3Seed3.pt",
    "./outputs/models/Model4Seed4.pt",
    "./outputs/models/Model5Seed5.pt",
    "./outputs/models/Model6Seed6.pt",
    "./outputs/models/Model7Seed7.pt"
]

ModelPaths = [p.replace("\\", "/") for p in ModelPaths]

UsePretrainedInitForTheseModels = True  # set False if your exp3 models were trained from scratch

for p in ModelPaths:
    print(p, "| exists:", os.path.exists(p))

ModelsEns = [LoadModelFromCheckpoint(p, UsePretrainedInit=UsePretrainedInitForTheseModels) for p in ModelPaths]
print("Loaded models:", len(ModelsEns))

BestSingleIdx = 0
for i, p in enumerate(ModelPaths):
    if "model6" in os.path.basename(p):
        BestSingleIdx = i
        break

BestModel = ModelsEns[BestSingleIdx]
print("Best single:", os.path.basename(ModelPaths[BestSingleIdx]))


# In[6]:


# Helpers: Prediction + Entropy + Ensemble + Rotation + Visualization
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

def EntropyFromProbs(Probs: torch.Tensor, Eps: float = 1e-8) -> torch.Tensor:
    """
    Predictive entropy to quantify uncertainty.
    """
    return -torch.sum(Probs * torch.log(Probs + Eps), dim=1)

def EnsemblePredictMajority(Models, BatchImage01: torch.Tensor) -> torch.Tensor:
    PredList = []
    for M in Models:
        Probs = PredictProbaOneModel(M, BatchImage01)
        PredList.append(Probs.argmax(dim=1))
    PredStack = torch.stack(PredList, dim=0)
    Voted, _ = torch.mode(PredStack, dim=0)
    return Voted

def EnsemblePredictProbAvgProbs(Models, BatchImage01: torch.Tensor) -> torch.Tensor:
    ProbsList = [PredictProbaOneModel(M, BatchImage01) for M in Models]
    ProbsStack = torch.stack(ProbsList, dim=0)
    AvgProbs = ProbsStack.mean(dim=0)
    return AvgProbs

def EnsemblePredictProbAvg(Models, BatchImage01: torch.Tensor) -> torch.Tensor:
    AvgProbs = EnsemblePredictProbAvgProbs(Models, BatchImage01)
    return AvgProbs.argmax(dim=1)

def RotateBatch(BatchImage01: torch.Tensor, AngleDeg: float) -> torch.Tensor:
    """
    I rotate each image in the batch by AngleDeg.
    """
    RotList = []
    for i in range(BatchImage01.shape[0]):
        RotImg = TF.rotate(BatchImage01[i], AngleDeg)
        RotList.append(RotImg)
    return torch.stack(RotList, dim=0)

def PlotGrid(Images01: torch.Tensor, YTrue: torch.Tensor, YPred: torch.Tensor, Title: str, SavePath: str = None) -> None:
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


# In[9]:


# Pick 20 random MNIST images (baseline)
SaveDirExp5 = "Exp5Results"
os.makedirs(SaveDirExp5, exist_ok=True)

IndexList = torch.randperm(len(TestDatasetNoNorm))[:20].tolist()
Samples = [TestDatasetNoNorm[i] for i in IndexList]

XClean = torch.stack([s[0] for s in Samples], dim=0)
YTrue  = torch.tensor([s[1] for s in Samples], dtype=torch.long)

# Clean single
ProbsSingleClean = PredictProbaOneModel(BestModel, XClean)
PredSingleClean = ProbsSingleClean.argmax(dim=1)
AccSingleClean = (PredSingleClean == YTrue).float().mean().item()
EntSingleClean = EntropyFromProbs(ProbsSingleClean).mean().item()

# Clean ensemble probavg
ProbsProbClean = EnsemblePredictProbAvgProbs(ModelsEns, XClean)
PredProbClean = ProbsProbClean.argmax(dim=1)
AccProbClean = (PredProbClean == YTrue).float().mean().item()
EntProbClean = EntropyFromProbs(ProbsProbClean).mean().item()

print(f"Clean Acc | Single={AccSingleClean*100:.2f}% | ProbAvg={AccProbClean*100:.2f}%")
print(f"Clean Entropy | Single={EntSingleClean:.4f} | ProbAvg={EntProbClean:.4f}")

PlotGrid(XClean, YTrue, PredSingleClean, "Exp5 Clean: Single", SavePath=os.path.join(SaveDirExp5, "Clean_Single.png"))
PlotGrid(XClean, YTrue, PredProbClean, "Exp5 Clean: ProbAvg", SavePath=os.path.join(SaveDirExp5, "Clean_ProbAvg.png"))


# In[10]:


# Rotation Experiment: 0° -> 180°
# Accuracy + Entropy for Single vs Ensemble ProbAvg
Angles = list(range(0, 181, 15))

AccSingleByAngle = []
AccProbByAngle = []

EntSingleByAngle = []
EntProbByAngle = []

for A in Angles:
    XRot = RotateBatch(XClean, A)

    # Single
    ProbsSingle = PredictProbaOneModel(BestModel, XRot)
    PredSingle = ProbsSingle.argmax(dim=1)
    AccSingle = (PredSingle == YTrue).float().mean().item()
    EntSingle = EntropyFromProbs(ProbsSingle).mean().item()

    # ProbAvg
    ProbsProb = EnsemblePredictProbAvgProbs(ModelsEns, XRot)
    PredProb = ProbsProb.argmax(dim=1)
    AccProb = (PredProb == YTrue).float().mean().item()
    EntProb = EntropyFromProbs(ProbsProb).mean().item()

    AccSingleByAngle.append(AccSingle)
    AccProbByAngle.append(AccProb)

    EntSingleByAngle.append(EntSingle)
    EntProbByAngle.append(EntProb)

    # Save grids for a few key angles only (to avoid too many files)
    if A in [0, 45, 90, 135, 180]:
        PlotGrid(XRot, YTrue, PredSingle,
                 f"Exp5: Single | Angle={A} | Acc={AccSingle*100:.1f}% | Ent={EntSingle:.3f}",
                 SavePath=os.path.join(SaveDirExp5, f"Angle_{A}_Single.png"))

        PlotGrid(XRot, YTrue, PredProb,
                 f"Exp5: ProbAvg | Angle={A} | Acc={AccProb*100:.1f}% | Ent={EntProb:.3f}",
                 SavePath=os.path.join(SaveDirExp5, f"Angle_{A}_ProbAvg.png"))


# In[11]:


# Save metrics
DfExp5 = pd.DataFrame({
    "AngleDeg": Angles,
    "AccSingle": AccSingleByAngle,
    "AccProbAvg": AccProbByAngle,
    "EntSingle": EntSingleByAngle,
    "EntProbAvg": EntProbByAngle
})

CsvPath = os.path.join(SaveDirExp5, "Exp5Metrics.csv")
DfExp5.to_csv(CsvPath, index=False)
print("Saved metrics:", CsvPath)

DfExp5


# In[12]:


# 1) Accuracy vs Angle
plt.figure(figsize=(7, 5))
plt.plot(Angles, AccSingleByAngle, marker="o", label="Single")
plt.plot(Angles, AccProbByAngle, marker="o", label="ProbAvg")
plt.xlabel("Rotation Angle (degrees)")
plt.ylabel("Accuracy (on 20 images)")
plt.title("Exp5: Accuracy vs Rotation Angle")
plt.grid(True)
plt.legend()
AccVsAnglePath = os.path.join(SaveDirExp5, "AccVsAngle.png")
plt.savefig(AccVsAnglePath, dpi=200, bbox_inches="tight")
plt.show()
print("Saved:", AccVsAnglePath)


# 2) Entropy vs Angle
plt.figure(figsize=(7, 5))
plt.plot(Angles, EntSingleByAngle, marker="o", label="Single")
plt.plot(Angles, EntProbByAngle, marker="o", label="ProbAvg")
plt.xlabel("Rotation Angle (degrees)")
plt.ylabel("Predictive Entropy")
plt.title("Exp5: Entropy vs Rotation Angle")
plt.grid(True)
plt.legend()
EntVsAnglePath = os.path.join(SaveDirExp5, "EntVsAngle.png")
plt.savefig(EntVsAnglePath, dpi=200, bbox_inches="tight")
plt.show()
print("Saved:", EntVsAnglePath)


# 3) Accuracy vs Entropy
plt.figure(figsize=(7, 5))
plt.scatter(EntSingleByAngle, AccSingleByAngle, label="Single")
plt.scatter(EntProbByAngle, AccProbByAngle, label="ProbAvg")
plt.xlabel("Predictive Entropy")
plt.ylabel("Accuracy (on 20 images)")
plt.title("Exp5: Accuracy vs Entropy")
plt.grid(True)
plt.legend()
AccVsEntropyPath = os.path.join(SaveDirExp5, "AccVsEntropy.png")
plt.savefig(AccVsEntropyPath, dpi=200, bbox_inches="tight")
plt.show()
print("Saved:", AccVsEntropyPath)


# **In Experiment 5, we studied the robustness of MNIST classifiers under image rotations between 0° and 180°. While the single ResNet18 model achieves perfect accuracy for unrotated digits, its performance degrades rapidly as rotation increases, accompanied by a rise in predictive entropy. Using an ensemble of seven independently trained models with probability averaging, we observe a consistently higher predictive entropy under distribution shift, reflecting increased epistemic uncertainty. Importantly, the ensemble’s uncertainty correlates with both visual ambiguity and accuracy degradation, demonstrating that ensemble-based uncertainty estimation provides a more reliable confidence measure than single-model predictions, especially in out-of-distribution settings.**

# In[ ]:




