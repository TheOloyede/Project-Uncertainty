#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import glob
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
from torchvision.models import resnet18

from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights


# In[2]:


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)


# In[3]:


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


# In[4]:


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


# In[5]:


DATA_DIR = "./data"  

weights = ResNet18_Weights.IMAGENET1K_V1
mean, std = weights.transforms().mean, weights.transforms().std
mean_t = torch.tensor(mean).view(3, 1, 1)
std_t  = torch.tensor(std).view(3, 1, 1)

test_transform_no_norm = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # [0,1]
])

test_dataset_no_norm = MNIST(root=DATA_DIR, train=False, download=True, transform=test_transform_no_norm)
print("Test size:", len(test_dataset_no_norm))

def normalize_for_model(x01: torch.Tensor) -> torch.Tensor:
    return (x01 - mean_t) / std_t


# In[ ]:


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATHS = [
    "outputs/models/exp1_scratch_best.pt",
    "outputs/models/exp1_pretrained_best.pt"
]

def load_model(path):
    m = resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 10)
    m.load_state_dict(torch.load(path, map_location=DEVICE))
    return m.to(DEVICE).eval()

models = [load_model(p) for p in MODEL_PATHS]
print("Loaded models:", len(models))


# In[ ]:


weights = ResNet18_Weights.IMAGENET1K_V1
mean, std = weights.transforms().mean, weights.transforms().std

test_tf = T.Compose([
    T.Grayscale(num_output_channels=3),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean, std),
])

test_ds = torchvision.datasets.MNIST(
    "./data", train=False, download=True, transform=test_tf
)
test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0)


# In[ ]:


class Args:
    dataDir = "./data"
    saveDir = "outputs/models"
    epochs = 10
    batchSize = 128
    learningRate = 0.001
    pretrained = False   
    seed = None

args = Args()


# In[ ]:


def trainModel(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    setSeed(args.seed)

    trainLoader, testLoader = getDataLoaders(
        args.dataDir, args.batchSize, device=device
    )

    model = createModel(pretrained=args.pretrained).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learningRate)

    bestAcc = 0.0
    bestPath = os.path.join(
        args.saveDir, f"exp3_scratch_seed{args.seed}_best.pt"
    )

    for epoch in range(args.epochs):
        trainLoss, trainAcc = trainOneEpoch(
            model, trainLoader, criterion, optimizer, device
        )
        testLoss, testAcc = evaluate(
            model, testLoader, criterion, device
        )

        print(
            f"Seed {args.seed} | Epoch {epoch+1} | "
            f"Train Acc {trainAcc:.2f}% | Test Acc {testAcc:.2f}%"
        )

        if testAcc > bestAcc:
            bestAcc = testAcc
            torch.save(model.state_dict(), bestPath)

    print(f"Saved best model → {bestPath}")


# In[ ]:


ensemble_info = []

seeds = [1, 2, 3, 4, 5, 6, 7]   # 7 different trainings
for i, s in enumerate(seeds, start=1):
    run_name, best_path, best_acc = train_one_model(model_id=i, seed=s, epochs=10, lr=1e-4)
    ensemble_info.append({
        "model_id": i,
        "seed": s,
        "run_name": run_name,
        "best_path": best_path,
        "best_acc": float(best_acc)
    })

# Saving the list of models 
with open(f"{EXP3_ROOT}/ensemble_models.json", "w") as f:
    json.dump(ensemble_info, f, indent=2)

ensemble_info


# In[9]:


CANDIDATE_DIRS = [
    "/content/outputs/models",   
    "./outputs/models",          
    "./models",                  
]

MODEL_DIR = None
for d in CANDIDATE_DIRS:
    if os.path.isdir(d):
        hits = glob.glob(os.path.join(d, "exp3_ens_model*_best.pt"))
        if len(hits) >= 7:
            MODEL_DIR = d
            break

if MODEL_DIR is None:
    raise FileNotFoundError(
        "Could not find a directory containing exp3_ens_model*_best.pt. "
        "Add your real folder path to CANDIDATE_DIRS."
    )

print("Using MODEL_DIR:", MODEL_DIR)

# Building the 7 model paths from the discovered directory
model_paths = sorted(glob.glob(os.path.join(MODEL_DIR, "exp3_ens_model*_best.pt")))
model_paths = model_paths[:7]  

print("Found paths:", len(model_paths))
for p in model_paths:
    print(" -", p, "| exists:", os.path.exists(p))

assert len(model_paths) == 7 and all(os.path.exists(p) for p in model_paths), "Model files missing!"


# In[11]:


model_paths = [p.replace("\\", "/") for p in model_paths]

print("Model paths:", len(model_paths))
for p in model_paths:
    print(" -", p, "| exists:", os.path.exists(p))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)


# In[12]:


PRETRAINED_INIT = True  

def build_resnet18(pretrained_init: bool):
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained_init else None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model.to(DEVICE)

def load_model_from_checkpoint(path: str, pretrained_init: bool):
    m = build_resnet18(pretrained_init)
    state = torch.load(path, map_location=DEVICE)
    m.load_state_dict(state)
    m.eval()
    return m

models_ens = [load_model_from_checkpoint(p, PRETRAINED_INIT) for p in model_paths]
print("Loaded", len(models_ens), "models.")


# In[13]:


from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import torch

DATA_DIR = "./data"  

weights = ResNet18_Weights.IMAGENET1K_V1
mean, std = weights.transforms().mean, weights.transforms().std
mean_t = torch.tensor(mean).view(3, 1, 1)
std_t  = torch.tensor(std).view(3, 1, 1)

test_transform_no_norm = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # [0,1]
])

test_dataset_no_norm = MNIST(root=DATA_DIR, train=False, download=True, transform=test_transform_no_norm)
print("Test size:", len(test_dataset_no_norm))

def normalize_for_model(x01: torch.Tensor) -> torch.Tensor:
    return (x01 - mean_t) / std_t


# In[14]:


import torch

idx = torch.randperm(len(test_dataset_no_norm))[:20].tolist()
samples = [test_dataset_no_norm[i] for i in idx]

x_clean = torch.stack([s[0] for s in samples], dim=0)  # [20,3,H,W] in [0,1]
y_true  = torch.tensor([s[1] for s in samples], dtype=torch.long)

print("Selected 20 random test images.")


# In[15]:


def predict_one_model(model, x01_batch: torch.Tensor) -> torch.Tensor:
    xb = torch.stack([normalize_for_model(x) for x in x01_batch], dim=0).to(DEVICE)
    with torch.no_grad():
        logits = model(xb)
        pred = logits.argmax(dim=1).cpu()
    return pred

def ensemble_predict_majority(models, x01_batch: torch.Tensor) -> torch.Tensor:
    preds = [predict_one_model(m, x01_batch) for m in models]  
    preds_stack = torch.stack(preds, dim=0)                   
    voted, _ = torch.mode(preds_stack, dim=0)                 # majority vote across models
    return voted


# In[18]:


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


# In[19]:


# Quick display test 
pred_test = y_true.clone()  # predictions just to test plotting
plot_grid(x_clean, y_true, pred_test, title="Plot test: grid display works")


# In[20]:


best_single_idx = 0
for i, p in enumerate(model_paths):
    if "model6" in os.path.basename(p):
        best_single_idx = i
        break

best_model = models_ens[best_single_idx]
print("Best single model:", os.path.basename(model_paths[best_single_idx]))

pred_single_clean = predict_one_model(best_model, x_clean)
acc_single_clean = (pred_single_clean == y_true).float().mean().item()

pred_ens_clean = ensemble_predict_majority(models_ens, x_clean)
acc_ens_clean = (pred_ens_clean == y_true).float().mean().item()

plot_grid(x_clean, y_true, pred_single_clean, title=f"Exp3: Best single model (clean) | Acc={acc_single_clean*100:.1f}%")
plot_grid(x_clean, y_true, pred_ens_clean, title=f"Exp3: Ensemble (7) majority vote (clean) | Acc={acc_ens_clean*100:.1f}%")

print("Clean Acc | Single:", acc_single_clean, "| Ensemble:", acc_ens_clean)


# In[21]:


noise_levels = [0.0, 0.05, 0.10, 0.20, 0.30]

acc_single = []
acc_ens = []

for lvl in noise_levels:
    noise = torch.randn_like(x_clean)
    x_noisy = (x_clean + noise * lvl).clamp(0, 1)  

    pred_single = predict_one_model(best_model, x_noisy)
    pred_ens = ensemble_predict_majority(models_ens, x_noisy)

    a1 = (pred_single == y_true).float().mean().item()
    aE = (pred_ens == y_true).float().mean().item()

    acc_single.append(a1)
    acc_ens.append(aE)

    plot_grid(x_noisy, y_true, pred_single, title=f"Exp3: Single model | noise={lvl:.2f} | Acc={a1*100:.1f}%")
    plot_grid(x_noisy, y_true, pred_ens, title=f"Exp3: Ensemble(7) majority vote | noise={lvl:.2f} | Acc={aE*100:.1f}%")

plt.figure()
plt.plot(noise_levels, acc_single, marker="o", label="Best single model")
plt.plot(noise_levels, acc_ens, marker="o", label="Ensemble (7) majority vote")
plt.xlabel("White noise level")
plt.ylabel("Accuracy on 20 images")
plt.title("Exp3: Robustness under noise — single vs ensemble")
plt.ylim(0, 1.05)
plt.grid(True)
plt.legend()
plt.show()


# 
# 
# ### **Benefit of using 7 models:**
# Majority voting reduces prediction variance because noise-induced errors are not perfectly correlated across independently trained networks. When one model makes a mistake under noise, others can still be correct, so the ensemble produces more stable and reliable predictions.

# ##  **Experiment 3 Analysis:**
# The results show that both the single ResNet18 model and the ensemble experience a degradation in accuracy as the white noise level increases. A noticeable visual degradation is not required to observe a significant drop in performance, since accuracy already decreases at moderate noise levels while digits remain partially recognizable.
# 
# The ensemble does not consistently outperform the best single model at all noise levels. At moderate noise intensities, the strongest individual model can achieve higher accuracy than the majority vote. This occurs because majority voting averages predictions from models with varying robustness, and errors made by weaker models can dominate the vote.
# 
# However, at very high noise levels, the ensemble exhibits improved stability and avoids complete performance collapse, outperforming the single model. This demonstrates that the benefit of using multiple models lies in variance reduction and robustness to extreme perturbations rather than guaranteed superiority over the best individual classifier.

# ## **Why ensemble is worse here?”**
# 
# Because majority voting reduces variance but can amplify bias when several weaker models fail in a correlated manner. In small-sample regimes, a strong single model can outperform a simple ensemble.

# In[22]:


import torch.nn.functional as F

def predict_proba_one_model(model, x01_batch: torch.Tensor) -> torch.Tensor:
    """
    Returns probabilities: shape [B,10]
    """
    xb = torch.stack([normalize_for_model(x) for x in x01_batch], dim=0).to(DEVICE)
    with torch.no_grad():
        logits = model(xb)                   # [B,10]
        probs = F.softmax(logits, dim=1)     # [B,10]
    return probs.cpu()

def ensemble_predict_proba_average(models, x01_batch: torch.Tensor) -> torch.Tensor:
    """
    Soft voting: average probabilities across models, then argmax.
    Returns predicted labels: [B]
    """
    probs_list = [predict_proba_one_model(m, x01_batch) for m in models]  
    probs_stack = torch.stack(probs_list, dim=0)                          
    avg_probs = probs_stack.mean(dim=0)                                   
    pred = avg_probs.argmax(dim=1)                                        
    return pred


# In[23]:


# Best single model selection
best_single_idx = 0
for i, p in enumerate(model_paths):
    if "model6" in os.path.basename(p):
        best_single_idx = i
        break

best_model = models_ens[best_single_idx]
print("Best single model:", os.path.basename(model_paths[best_single_idx]))

# Single model
pred_single_clean = predict_one_model(best_model, x_clean)
acc_single_clean = (pred_single_clean == y_true).float().mean().item()

# Majority vote ensemble
pred_maj_clean = ensemble_predict_majority(models_ens, x_clean)
acc_maj_clean = (pred_maj_clean == y_true).float().mean().item()

# Probability averaging ensemble 
pred_soft_clean = ensemble_predict_proba_average(models_ens, x_clean)
acc_soft_clean = (pred_soft_clean == y_true).float().mean().item()

plot_grid(x_clean, y_true, pred_single_clean, title=f"Exp3: Best single model (clean) | Acc={acc_single_clean*100:.1f}%")
plot_grid(x_clean, y_true, pred_maj_clean, title=f"Exp3: Ensemble majority vote (clean) | Acc={acc_maj_clean*100:.1f}%")
plot_grid(x_clean, y_true, pred_soft_clean, title=f"Exp3: Ensemble prob-avg (clean) | Acc={acc_soft_clean*100:.1f}%")

print("Clean Acc | Single:", acc_single_clean, "| Majority:", acc_maj_clean, "| Prob-Avg:", acc_soft_clean)


# In[24]:


noise_levels = [0.0, 0.05, 0.10, 0.20, 0.30]

acc_single = []
acc_maj = []
acc_soft = []

for lvl in noise_levels:
    noise = torch.randn_like(x_clean)
    x_noisy = (x_clean + noise * lvl).clamp(0, 1)

    pred_single = predict_one_model(best_model, x_noisy)
    pred_maj = ensemble_predict_majority(models_ens, x_noisy)
    pred_soft = ensemble_predict_proba_average(models_ens, x_noisy)

    a1 = (pred_single == y_true).float().mean().item()
    aM = (pred_maj == y_true).float().mean().item()
    aS = (pred_soft == y_true).float().mean().item()

    acc_single.append(a1)
    acc_maj.append(aM)
    acc_soft.append(aS)

    plot_grid(x_noisy, y_true, pred_single, title=f"Exp3: Single | noise={lvl:.2f} | Acc={a1*100:.1f}%")
    plot_grid(x_noisy, y_true, pred_maj, title=f"Exp3: Majority vote | noise={lvl:.2f} | Acc={aM*100:.1f}%")
    plot_grid(x_noisy, y_true, pred_soft, title=f"Exp3: Prob-avg | noise={lvl:.2f} | Acc={aS*100:.1f}%")

# Comparison plot
plt.figure()
plt.plot(noise_levels, acc_single, marker="o", label="Best single model")
plt.plot(noise_levels, acc_maj, marker="o", label="Ensemble majority vote")
plt.plot(noise_levels, acc_soft, marker="o", label="Ensemble probability averaging")
plt.xlabel("White noise level")
plt.ylabel("Accuracy on 20 images")
plt.title("Exp3: Noise robustness — single vs ensemble (majority vs prob-avg)")
plt.ylim(0, 1.05)
plt.grid(True)
plt.legend()
plt.show()


# ## Experiment 3 (Ensemble Robustness):
# 
# The results show that classification accuracy decreases with increasing white noise for all methods.
# 
# Comparing ensemble strategies, probability averaging produces a smoother and more stable degradation curve than majority voting, as it mitigates the influence of weak models by weighting predictions according to confidence. However, the strongest individual model can still outperform the ensemble at certain noise levels, particularly when the evaluation set is small.
# 
# Overall, the benefit of using multiple models lies in improved robustness and variance reduction rather than guaranteed superiority over the best single classifier.

# In[25]:


# Changing evaluation set size from 20 to 200

N_EVAL = 200

idx = torch.randperm(len(test_dataset_no_norm))[:N_EVAL].tolist()
samples = [test_dataset_no_norm[i] for i in idx]

x_eval = torch.stack([s[0] for s in samples], dim=0)   
y_eval = torch.tensor([s[1] for s in samples], dtype=torch.long)

print("Eval set size:", len(x_eval))


# In[26]:


import torch.nn.functional as F
import numpy as np

def predict_proba_one_model(model, x01_batch: torch.Tensor) -> torch.Tensor:
    xb = torch.stack([normalize_for_model(x) for x in x01_batch], dim=0).to(DEVICE)
    with torch.no_grad():
        logits = model(xb)
        probs = F.softmax(logits, dim=1)
    return probs.cpu()  

def entropy_from_probs(probs: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    probs: [B,C]
    returns: [B] entropy in nats
    """
    p = probs.clamp(min=eps)
    return -(p * p.log()).sum(dim=1)


# In[27]:


noise_levels = [0.0, 0.05, 0.10, 0.20, 0.30]

acc_per_model = np.zeros((len(models_ens), len(noise_levels)), dtype=float)

for j, lvl in enumerate(noise_levels):
    noise = torch.randn_like(x_eval)
    x_noisy = (x_eval + noise * lvl).clamp(0, 1)

    for i, m in enumerate(models_ens):
        probs = predict_proba_one_model(m, x_noisy)          
        pred = probs.argmax(dim=1)                           
        acc = (pred == y_eval).float().mean().item()
        acc_per_model[i, j] = acc

acc_per_model


plt.figure(figsize=(8,5))
for i in range(acc_per_model.shape[0]):
    label = os.path.basename(model_paths[i]).replace("_best.pt","")
    plt.plot(noise_levels, acc_per_model[i], marker="o", alpha=0.8, label=label)

plt.xlabel("White noise level")
plt.ylabel(f"Accuracy on {N_EVAL} images")
plt.title("Per-model accuracy vs noise (7 models) — diversity check")
plt.ylim(0, 1.05)
plt.grid(True)
plt.legend(fontsize=8)
plt.show()

# In[29]:


def ensemble_predict_majority(models, x01_batch: torch.Tensor) -> torch.Tensor:
    preds = []
    for m in models:
        probs = predict_proba_one_model(m, x01_batch)
        preds.append(probs.argmax(dim=1))
    preds_stack = torch.stack(preds, dim=0)     
    voted, _ = torch.mode(preds_stack, dim=0)   
    return voted

def ensemble_predict_probavg(models, x01_batch: torch.Tensor) -> torch.Tensor:
    probs_list = [predict_proba_one_model(m, x01_batch) for m in models]  
    probs_stack = torch.stack(probs_list, dim=0)
    avg_probs = probs_stack.mean(dim=0)                                  
    return avg_probs.argmax(dim=1)                                       

# picking the best single model
best_single_idx = 0
for i, p in enumerate(model_paths):
    if "model6" in os.path.basename(p):
        best_single_idx = i
        break
best_model = models_ens[best_single_idx]
print("Best single:", os.path.basename(model_paths[best_single_idx]))

acc_single = []
acc_maj = []
acc_prob = []

for lvl in noise_levels:
    noise = torch.randn_like(x_eval)
    x_noisy = (x_eval + noise * lvl).clamp(0, 1)

    # single
    pred_single = predict_proba_one_model(best_model, x_noisy).argmax(dim=1)
    acc_single.append((pred_single == y_eval).float().mean().item())

    # majority
    pred_maj = ensemble_predict_majority(models_ens, x_noisy)
    acc_maj.append((pred_maj == y_eval).float().mean().item())

    # prob-avg
    pred_prob = ensemble_predict_probavg(models_ens, x_noisy)
    acc_prob.append((pred_prob == y_eval).float().mean().item())

plt.figure(figsize=(7,5))
plt.plot(noise_levels, acc_single, marker="o", label="Best single model")
plt.plot(noise_levels, acc_maj, marker="o", label="Ensemble majority vote")
plt.plot(noise_levels, acc_prob, marker="o", label="Ensemble probability averaging")
plt.xlabel("White noise level")
plt.ylabel(f"Accuracy on {N_EVAL} images")
plt.title("Exp3: Noise robustness (200 images)")
plt.ylim(0, 1.05)
plt.grid(True)
plt.legend()
plt.show()

acc_single, acc_maj, acc_prob


# In[30]:


entropy_per_model = np.zeros((len(models_ens), len(noise_levels)), dtype=float)
entropy_ens_probavg = np.zeros(len(noise_levels), dtype=float)

for j, lvl in enumerate(noise_levels):
    noise = torch.randn_like(x_eval)
    x_noisy = (x_eval + noise * lvl).clamp(0, 1)

    # per-model entropy
    probs_list = []
    for i, m in enumerate(models_ens):
        probs = predict_proba_one_model(m, x_noisy)                 
        probs_list.append(probs)
        ent = entropy_from_probs(probs).mean().item()               # mean entropy over images
        entropy_per_model[i, j] = ent

    # ensemble prob-avg entropy (average probs then entropy)
    probs_stack = torch.stack(probs_list, dim=0)                    
    avg_probs = probs_stack.mean(dim=0)                             
    entropy_ens_probavg[j] = entropy_from_probs(avg_probs).mean().item()

entropy_per_model, entropy_ens_probavg


# In[31]:


plt.figure(figsize=(8,5))
for i in range(entropy_per_model.shape[0]):
    label = os.path.basename(model_paths[i]).replace("_best.pt","")
    plt.plot(noise_levels, entropy_per_model[i], marker="o", alpha=0.8)

plt.plot(noise_levels, entropy_ens_probavg, marker="o", linewidth=3, label="Ensemble prob-avg (mean entropy)")
plt.xlabel("White noise level")
plt.ylabel("Mean predictive entropy (nats)")
plt.title(f"Confidence vs noise (Entropy) — {N_EVAL} images")
plt.grid(True)
plt.legend()
plt.show()


# In[32]:


plt.figure(figsize=(6,5))
plt.scatter(entropy_ens_probavg, acc_prob)

for j, lvl in enumerate(noise_levels):
    plt.annotate(f"{lvl:.2f}", (entropy_ens_probavg[j], acc_prob[j]))

plt.xlabel("Ensemble mean entropy (nats)")
plt.ylabel(f"Ensemble accuracy on {N_EVAL} images")
plt.title("Accuracy vs uncertainty (ensemble prob-avg)")
plt.grid(True)
plt.show()


# ### **Ensemble Robustness and Uncertainty Analysis:**
# 
# The per-model accuracy analysis reveals significant diversity in robustness among independently trained ResNet18 models, with different models exhibiting distinct degradation rates under increasing white noise. This confirms that ensemble learning is meaningful in this setting, as model errors are not perfectly correlated.
# 
# When evaluated on a larger test set of 200 images, both majority voting and probability averaging produce smoother degradation curves than a single classifier, demonstrating improved stability. However, the strongest individual model can still outperform the ensemble at moderate noise levels, highlighting that ensembles primarily reduce variance rather than guaranteeing superior peak accuracy.
# 
# Confidence analysis using predictive entropy shows a clear increase in uncertainty as noise intensity rises. This increase precedes severe accuracy degradation, indicating that uncertainty is a sensitive indicator of robustness loss. The relationship between entropy and accuracy further confirms that performance degradation under noise is strongly linked to increasing model uncertainty.
# 
# Overall, these results demonstrate that noticeable visual degradation is not required to observe a significant drop in classification performance, and that ensemble methods improve robustness by stabilizing predictions and uncertainty rather than consistently outperforming the best individual model.

# In[ ]:




