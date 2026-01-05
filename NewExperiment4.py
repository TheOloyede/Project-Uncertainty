#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import ssl
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision.datasets import MNIST, KMNIST
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights


# In[2]:


# Device and Reproducibility
Device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", Device)

def SeedEverything(SeedValue: int = 42) -> None:
    random.seed(SeedValue)
    np.random.seed(SeedValue)
    torch.manual_seed(SeedValue)
    torch.cuda.manual_seed_all(SeedValue)




# In[18]:


# Model Construction / Loading
def BuildResNet18(UsePretrainedInit: bool) -> torch.nn.Module:
    Weights = ResNet18_Weights.IMAGENET1K_V1 if UsePretrainedInit else None
    Model = resnet18(weights=Weights)
    Model.fc = nn.Linear(Model.fc.in_features, 10)
    return Model.to(Device)

def LoadModelFromCheckpoint(CheckpointPath: str, UsePretrainedInit: bool) -> torch.nn.Module:
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
    Returns ImageNet-normalized tensor (required for ResNet18 pretrained).
    """
    return (Image01 - MeanTensor) / StdTensor

TestTransformNoNorm = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# In[5]:


# KMNIST Download SSL Workaround
try:
    _ = ssl._create_default_https_context
    ssl._create_default_https_context = ssl._create_unverified_context
    print("SSL context patched (if needed).")
except Exception as Err:
    print("SSL patch not applied:", Err)


# In[6]:


# Load datasets
MnistTestNoNorm = MNIST(root=DataDir, train=False, download=True, transform=TestTransformNoNorm)
KmnistTestNoNorm = KMNIST(root=DataDir, train=False, download=True, transform=TestTransformNoNorm)

print("MNIST test:", len(MnistTestNoNorm))
print("KMNIST test:", len(KmnistTestNoNorm))


# In[19]:


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

# Best single model:
BestSingleIdx = 0
for i, p in enumerate(ModelPaths):
    if "model6" in os.path.basename(p):
        BestSingleIdx = i
        break
BestModel = ModelsEns[BestSingleIdx]
print("Best single:", os.path.basename(ModelPaths[BestSingleIdx]))


# In[11]:


# Helpers: Prediction + Entropy + Ensemble + Visualization
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
    Higher entropy => less confident.
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



# In[12]:


# Sample selection: 20 MNIST + 20 KMNIST
SaveDirExp4 = "Exp4Results"
os.makedirs(SaveDirExp4, exist_ok=True)

MnistIdx = torch.randperm(len(MnistTestNoNorm))[:20].tolist()
KmnistIdx = torch.randperm(len(KmnistTestNoNorm))[:20].tolist()

MnistSamples = [MnistTestNoNorm[i] for i in MnistIdx]
KmnistSamples = [KmnistTestNoNorm[i] for i in KmnistIdx]

XMnist = torch.stack([s[0] for s in MnistSamples], dim=0)
YMnist = torch.tensor([s[1] for s in MnistSamples], dtype=torch.long)

XKmnist = torch.stack([s[0] for s in KmnistSamples], dim=0)
YKmnist = torch.tensor([s[1] for s in KmnistSamples], dtype=torch.long)

print("XMnist:", XMnist.shape, "XKmnist:", XKmnist.shape)


# In[13]:


# Evaluation: MNIST vs KMNIST
# Accuracy + Entropy (Single + Ensemble)
# MNIST
ProbsSingleMn = PredictProbaOneModel(BestModel, XMnist)
PredSingleMn = ProbsSingleMn.argmax(dim=1)
AccSingleMn = (PredSingleMn == YMnist).float().mean().item()
EntSingleMn = EntropyFromProbs(ProbsSingleMn).mean().item()

PredMajMn = EnsemblePredictMajority(ModelsEns, XMnist)
AccMajMn = (PredMajMn == YMnist).float().mean().item()

ProbsProbMn = EnsemblePredictProbAvgProbs(ModelsEns, XMnist)
PredProbMn = ProbsProbMn.argmax(dim=1)
AccProbMn = (PredProbMn == YMnist).float().mean().item()
EntProbMn = EntropyFromProbs(ProbsProbMn).mean().item()

# ---- KMNIST ----
ProbsSingleKm = PredictProbaOneModel(BestModel, XKmnist)
PredSingleKm = ProbsSingleKm.argmax(dim=1)
AccSingleKm = (PredSingleKm == YKmnist).float().mean().item()
EntSingleKm = EntropyFromProbs(ProbsSingleKm).mean().item()

PredMajKm = EnsemblePredictMajority(ModelsEns, XKmnist)
AccMajKm = (PredMajKm == YKmnist).float().mean().item()

ProbsProbKm = EnsemblePredictProbAvgProbs(ModelsEns, XKmnist)
PredProbKm = ProbsProbKm.argmax(dim=1)
AccProbKm = (PredProbKm == YKmnist).float().mean().item()
EntProbKm = EntropyFromProbs(ProbsProbKm).mean().item()

print("MNIST  | Acc Single:", AccSingleMn, "| Acc Maj:", AccMajMn, "| Acc Prob:", AccProbMn, "| Ent Single:", EntSingleMn, "| Ent Prob:", EntProbMn)
print("KMNIST | Acc Single:", AccSingleKm, "| Acc Maj:", AccMajKm, "| Acc Prob:", AccProbKm, "| Ent Single:", EntSingleKm, "| Ent Prob:", EntProbKm)


# In[14]:


# Save image grids for report
# MNIST grids
PlotGrid(XMnist, YMnist, PredSingleMn, "Exp4 MNIST: Single", SavePath=os.path.join(SaveDirExp4, "MNIST_Single.png"))
PlotGrid(XMnist, YMnist, PredMajMn,   "Exp4 MNIST: Majority", SavePath=os.path.join(SaveDirExp4, "MNIST_Majority.png"))
PlotGrid(XMnist, YMnist, PredProbMn,  "Exp4 MNIST: ProbAvg", SavePath=os.path.join(SaveDirExp4, "MNIST_ProbAvg.png"))

# KMNIST grids
PlotGrid(XKmnist, YKmnist, PredSingleKm, "Exp4 KMNIST: Single", SavePath=os.path.join(SaveDirExp4, "KMNIST_Single.png"))
PlotGrid(XKmnist, YKmnist, PredMajKm,    "Exp4 KMNIST: Majority", SavePath=os.path.join(SaveDirExp4, "KMNIST_Majority.png"))
PlotGrid(XKmnist, YKmnist, PredProbKm,   "Exp4 KMNIST: ProbAvg", SavePath=os.path.join(SaveDirExp4, "KMNIST_ProbAvg.png"))


# In[15]:


# Save metrics to CSV
DfExp4 = pd.DataFrame({
    "Dataset": ["MNIST", "KMNIST"],

    "AccSingle": [AccSingleMn, AccSingleKm],
    "AccMajority": [AccMajMn, AccMajKm],
    "AccProbAvg": [AccProbMn, AccProbKm],

    "EntSingle": [EntSingleMn, EntSingleKm],
    "EntProbAvg": [EntProbMn, EntProbKm]
})

CsvPath = os.path.join(SaveDirExp4, "Exp4Metrics.csv")
DfExp4.to_csv(CsvPath, index=False)
print("Saved metrics:", CsvPath)

DfExp4


# In[17]:


# Exp4: Separate Plots

# 1) Accuracy comparison (bar)
plt.figure(figsize=(7, 5))
XPos = np.arange(2)
plt.bar(XPos - 0.2, [AccSingleMn, AccSingleKm], width=0.2, label="Single")
plt.bar(XPos,       [AccMajMn, AccMajKm],       width=0.2, label="Majority")
plt.bar(XPos + 0.2, [AccProbMn, AccProbKm],     width=0.2, label="ProbAvg")
plt.xticks(XPos, ["MNIST", "KMNIST"])
plt.ylabel("Accuracy (on 20 images)")
plt.title("Exp4: Accuracy — In-Distribution vs OOD")
plt.grid(True, axis="y")
plt.legend()

AccBarPath = os.path.join(SaveDirExp4, "AccVsDataset.png")
plt.savefig(AccBarPath, dpi=200, bbox_inches="tight")
plt.show()
print("Saved:", AccBarPath)


# 2) Entropy comparison (bar)
plt.figure(figsize=(7, 5))
plt.bar(XPos - 0.15, [EntSingleMn, EntSingleKm], width=0.3, label="Single")
plt.bar(XPos + 0.15, [EntProbMn, EntProbKm],     width=0.3, label="ProbAvg")
plt.xticks(XPos, ["MNIST", "KMNIST"])
plt.ylabel("Predictive Entropy")
plt.title("Exp4: Entropy — In-Distribution vs OOD")
plt.grid(True, axis="y")
plt.legend()

EntBarPath = os.path.join(SaveDirExp4, "EntVsDataset.png")
plt.savefig(EntBarPath, dpi=200, bbox_inches="tight")
plt.show()
print("Saved:", EntBarPath)


AccEntPath = os.path.join(SaveDirExp4, "AccVsEntropy.png")
plt.savefig(AccEntPath, dpi=200, bbox_inches="tight")
plt.show()
print("Saved:", AccEntPath)


# In[ ]:




