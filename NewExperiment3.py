#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing all libraries
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

TestDatasetNoNorm = MNIST(root=DataDir, train=False, download=True, transform=TestTransformNoNorm)
print("MNIST Test Size:", len(TestDatasetNoNorm))


# In[8]:


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

UsePretrainedInitForTheseModels = True  # Set False if your Exp3 models were trained from scratch without ImageNet init

for p in ModelPaths:
    print(p, "| exists:", os.path.exists(p))

ModelsEns = [LoadModelFromCheckpoint(p, UsePretrainedInit=UsePretrainedInitForTheseModels) for p in ModelPaths]
print("Loaded models:", len(ModelsEns))


# In[9]:


#Prediction + Entropy + Ensemble + Visualization
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
    I use predictive entropy to quantify uncertainty.
    Higher entropy => less confident predictions.
    """
    return -torch.sum(Probs * torch.log(Probs + Eps), dim=1)

def EnsemblePredictMajority(Models, BatchImage01: torch.Tensor) -> torch.Tensor:
    """
    Majority vote based on argmax predictions.
    """
    PredList = []
    for M in Models:
        Probs = PredictProbaOneModel(M, BatchImage01)
        PredList.append(Probs.argmax(dim=1))
    PredStack = torch.stack(PredList, dim=0)   # [K,B]
    Voted, _ = torch.mode(PredStack, dim=0)    # [B]
    return Voted

def EnsemblePredictProbAvgProbs(Models, BatchImage01: torch.Tensor) -> torch.Tensor:
    """
    Probability averaging: average softmax probabilities across models.
    """
    ProbsList = [PredictProbaOneModel(M, BatchImage01) for M in Models]  # list of [B,10]
    ProbsStack = torch.stack(ProbsList, dim=0)                           # [K,B,10]
    AvgProbs = ProbsStack.mean(dim=0)                                    # [B,10]
    return AvgProbs

def EnsemblePredictProbAvg(Models, BatchImage01: torch.Tensor) -> torch.Tensor:
    """
    Final class = argmax of averaged probabilities.
    """
    AvgProbs = EnsemblePredictProbAvgProbs(Models, BatchImage01)
    return AvgProbs.argmax(dim=1)




# In[10]:


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


# In[11]:


# Clean Evaluation (20 random images)
SaveDirExp3 = "Exp3Results"
os.makedirs(SaveDirExp3, exist_ok=True)

IndexList = torch.randperm(len(TestDatasetNoNorm))[:20].tolist()
Samples = [TestDatasetNoNorm[i] for i in IndexList]

XClean = torch.stack([s[0] for s in Samples], dim=0)   # [20,3,H,W]
YTrue  = torch.tensor([s[1] for s in Samples], dtype=torch.long)

# Best single model 
BestSingleIdx = 0
for i, p in enumerate(ModelPaths):
    if "model6" in os.path.basename(p):
        BestSingleIdx = i
        break

BestModel = ModelsEns[BestSingleIdx]
print("Best single model:", os.path.basename(ModelPaths[BestSingleIdx]))

# Single
ProbsSingleClean = PredictProbaOneModel(BestModel, XClean)
PredSingleClean = ProbsSingleClean.argmax(dim=1)
AccSingleClean = (PredSingleClean == YTrue).float().mean().item()
EntSingleClean = EntropyFromProbs(ProbsSingleClean).mean().item()

# Majority
PredMajClean = EnsemblePredictMajority(ModelsEns, XClean)
AccMajClean = (PredMajClean == YTrue).float().mean().item()

# Prob-Avg
ProbsProbClean = EnsemblePredictProbAvgProbs(ModelsEns, XClean)
PredProbClean = ProbsProbClean.argmax(dim=1)
AccProbClean = (PredProbClean == YTrue).float().mean().item()
EntProbClean = EntropyFromProbs(ProbsProbClean).mean().item()

print(f"Clean Acc | Single={AccSingleClean*100:.2f}% | Majority={AccMajClean*100:.2f}% | ProbAvg={AccProbClean*100:.2f}%")
print(f"Clean Entropy | Single={EntSingleClean:.4f} | ProbAvg={EntProbClean:.4f}")

PlotGrid(XClean, YTrue, PredSingleClean,
         Title="Exp3 Clean: Best Single Model",
         SavePath=os.path.join(SaveDirExp3, "Clean_Single.png"))

PlotGrid(XClean, YTrue, PredMajClean,
         Title="Exp3 Clean: Ensemble Majority Vote",
         SavePath=os.path.join(SaveDirExp3, "Clean_Majority.png"))

PlotGrid(XClean, YTrue, PredProbClean,
         Title="Exp3 Clean: Ensemble Probability Averaging",
         SavePath=os.path.join(SaveDirExp3, "Clean_ProbAvg.png"))


# In[15]:


# Noise Robustness + Uncertainty (Accuracy + Entropy)
# (Single vs Majority vs ProbAvg)
NoiseLevels = [0.0, 0.05, 0.10, 0.20, 0.30]

SingleByNoise = []
MajByNoise = []
ProbByNoise = []

EntSingleByNoise = []
EntProbByNoise = []  # entropy is meaningful for prob-avg (distribution). Majority is hard labels.

for Lvl in NoiseLevels:
    Noise = torch.randn_like(XClean)
    XNoisy = (XClean + Noise * Lvl).clamp(0, 1)

    # Single
    ProbsSingle = PredictProbaOneModel(BestModel, XNoisy)
    PredSingle = ProbsSingle.argmax(dim=1)
    AccSingle = (PredSingle == YTrue).float().mean().item()
    EntSingle = EntropyFromProbs(ProbsSingle).mean().item()

    # Majority
    PredMaj = EnsemblePredictMajority(ModelsEns, XNoisy)
    AccMaj = (PredMaj == YTrue).float().mean().item()

    # ProbAvg
    ProbsProb = EnsemblePredictProbAvgProbs(ModelsEns, XNoisy)
    PredProb = ProbsProb.argmax(dim=1)
    AccProb = (PredProb == YTrue).float().mean().item()
    EntProb = EntropyFromProbs(ProbsProb).mean().item()

    SingleByNoise.append(AccSingle)
    MajByNoise.append(AccMaj)
    ProbByNoise.append(AccProb)

    EntSingleByNoise.append(EntSingle)
    EntProbByNoise.append(EntProb)

    # Save 3 grids per noise level
    PlotGrid(XNoisy, YTrue, PredSingle,
             Title=f"Exp3: Single | Noise={Lvl:.2f} | Acc={AccSingle*100:.1f}% | Ent={EntSingle:.3f}",
             SavePath=os.path.join(SaveDirExp3, f"Noise_{Lvl:.2f}_Single.png"))

    PlotGrid(XNoisy, YTrue, PredMaj,
             Title=f"Exp3: Majority | Noise={Lvl:.2f} | Acc={AccMaj*100:.1f}%",
             SavePath=os.path.join(SaveDirExp3, f"Noise_{Lvl:.2f}_Majority.png"))

    PlotGrid(XNoisy, YTrue, PredProb,
             Title=f"Exp3: ProbAvg | Noise={Lvl:.2f} | Acc={AccProb*100:.1f}% | Ent={EntProb:.3f}",
             SavePath=os.path.join(SaveDirExp3, f"Noise_{Lvl:.2f}_ProbAvg.png"))


# In[16]:


# =========================
# Save Metrics
# =========================

DfExp3 = pd.DataFrame({
    "NoiseLevel": NoiseLevels,

    "Single": SingleByNoise,
    "Majority": MajByNoise,
    "ProbAvg": ProbByNoise,
vw
    "EntSingle": EntSingleByNoise,
    "EntProbAvg": EntProbByNoise
})

CsvPath = os.path.join(SaveDirExp3, "Exp3Metrics.csv")
DfExp3.to_csv(CsvPath, index=False)
print("Saved metrics:", CsvPath)

DfExp3


# In[19]:


# =========================
# Exp3 — Separate Plots (saved)
# =========================

# 1) Accuracy vs Noise
plt.figure(figsize=(7, 5))
plt.plot(NoiseLevels, SingleByNoise, marker="o", label="Single")
plt.plot(NoiseLevels, MajByNoise, marker="o", label="Majority")
plt.plot(NoiseLevels, ProbByNoise, marker="o", label="ProbAvg")
plt.xlabel("White Noise Level")
plt.ylabel("Accuracy (on 20 images)")
plt.title("Exp3: Accuracy vs Noise Level")
plt.grid(True)
plt.legend()
AccNoisePath = os.path.join(SaveDirExp3, "AccVsNoise.png")
plt.savefig(AccNoisePath, dpi=200, bbox_inches="tight")
plt.show()
print("Saved:", AccNoisePath)


# 2) Entropy vs Noise (Single vs ProbAvg)
plt.figure(figsize=(7, 5))
plt.plot(NoiseLevels, EntSingleByNoise, marker="o", label="Single")
plt.plot(NoiseLevels, EntProbByNoise, marker="o", label="ProbAvg")
plt.xlabel("White Noise Level")
plt.ylabel("Predictive Entropy")
plt.title("Exp3: Entropy vs Noise Level")
plt.grid(True)
plt.legend()
EntNoisePath = os.path.join(SaveDirExp3, "EntVsNoise.png")
plt.savefig(EntNoisePath, dpi=200, bbox_inches="tight")
plt.show()
print("Saved:", EntNoisePath)


# 3) Accuracy vs Entropy (scatter is best scientifically)
plt.figure(figsize=(7, 5))
plt.scatter(EntSingleByNoise, SingleByNoise, label="Single")
plt.scatter(EntProbByNoise, ProbByNoise, label="ProbAvg")
plt.xlabel("Predictive Entropy")
plt.ylabel("Accuracy (on 20 images)")
plt.title("Exp3: Accuracy vs Entropy")
plt.grid(True)
plt.legend()
AccEntPath = os.path.join(SaveDirExp3, "AccVsEntropy.png")
plt.savefig(AccEntPath, dpi=200, bbox_inches="tight")
plt.show()
print("Saved:", AccEntPath)


# ### Benefit of using 7 models:
# Majority voting reduces prediction variance because noise-induced errors are not perfectly correlated across independently trained networks. When one model makes a mistake under noise, others can still be correct, so the ensemble produces more stable and reliable predictions.

# ### Experiment 3 Analysis:
# The results show that both the single ResNet18 model and the ensemble experience a degradation in accuracy as the white noise level increases. A noticeable visual degradation is not required to observe a significant drop in performance, since accuracy already decreases at moderate noise levels while digits remain partially recognizable.
# 
# The ensemble does not consistently outperform the best single model at all noise levels. At moderate noise intensities, the strongest individual model can achieve higher accuracy than the majority vote. This occurs because majority voting averages predictions from models with varying robustness, and errors made by weaker models can dominate the vote.
# 
# However, at very high noise levels, the ensemble exhibits improved stability and avoids complete performance collapse, outperforming the single model. This demonstrates that the benefit of using multiple models lies in variance reduction and robustness to extreme perturbations rather than guaranteed superiority over the best individual classifier.

# ### **Why ensemble is worse here?”**
# Because majority voting reduces variance but can amplify bias when several weaker models fail in a correlated manner. In small-sample regimes, a strong single model can outperform a simple ensemble.

# ### **Experiment 3 (Ensemble Robustness):**
# The results show that classification accuracy decreases with increasing white noise for all methods.
# 
# Comparing ensemble strategies, probability averaging produces a smoother and more stable degradation curve than majority voting, as it mitigates the influence of weak models by weighting predictions according to confidence. However, the strongest individual model can still outperform the ensemble at certain noise levels, particularly when the evaluation set is small.
# 
# Overall, the benefit of using multiple models lies in improved robustness and variance reduction rather than guaranteed superiority over the best single classifier.

# Confidence analysis using predictive entropy shows a clear increase in uncertainty as noise intensity rises. This increase precedes severe accuracy degradation, indicating that uncertainty is a sensitive indicator of robustness loss. The relationship between entropy and accuracy further confirms that performance degradation under noise is strongly linked to increasing model uncertainty.
# 
# Overall, these results demonstrate that noticeable visual degradation is not required to observe a significant drop in classification performance, and that ensemble methods improve robustness by stabilizing predictions and uncertainty rather than consistently outperforming the best individual model.

# In[ ]:




