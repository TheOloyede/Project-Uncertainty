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
import torchvision.transforms.functional as TF
from torchvision.datasets import MNIST
import torchvision.transforms as transforms


# In[2]:


DATA_DIR = "./data"

test_transform_no_norm = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # [0,1]
])

mnist_test = MNIST(root=DATA_DIR, train=False, download=True, transform=test_transform_no_norm)
print("MNIST test:", len(mnist_test))


# In[3]:


N_SHOW = 20
idx = torch.randperm(len(mnist_test))[:N_SHOW].tolist()
samples = [mnist_test[i] for i in idx]

x0 = torch.stack([s[0] for s in samples], dim=0)   # [20,3,224,224] in [0,1]
y0 = torch.tensor([s[1] for s in samples], dtype=torch.long)

print("Batch:", x0.shape, y0.shape)


# In[11]:


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

PRETRAINED_INIT = True  # matches Exp3 training

def build_resnet18(pretrained_init=True):
    model = resnet18(
        weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained_init else None
    )
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model.to(DEVICE)

def load_model(path):
    model = build_resnet18(PRETRAINED_INIT)
    state = torch.load(path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model

# Load ensemble
models_ens = [load_model(p) for p in model_paths]
print("Loaded models:", len(models_ens))


# Select best single model
best_single_idx = 5  # model6_seed6_best.pt
best_model = models_ens[best_single_idx]
print("Best single model:", os.path.basename(model_paths[best_single_idx]))



# In[12]:


# picking the best single model 
best_single_idx = 0
for i, p in enumerate(model_paths):
    if "model6" in os.path.basename(p):
        best_single_idx = i
        break
best_model = models_ens[best_single_idx]
print("Best single model:", os.path.basename(model_paths[best_single_idx]))

def rotate_batch(x01_batch: torch.Tensor, angle_deg: float) -> torch.Tensor:
    """
    x01_batch: [B,3,H,W] in [0,1]
    rotate each image by angle_deg
    """
    out = []
    for i in range(x01_batch.shape[0]):
        out.append(TF.rotate(x01_batch[i], angle_deg, interpolation=TF.InterpolationMode.BILINEAR))
    return torch.stack(out, dim=0).clamp(0, 1)

def predict_proba_one_model(model, x01_batch: torch.Tensor) -> torch.Tensor:
    xb = torch.stack([normalize_for_model(x) for x in x01_batch], dim=0).to(DEVICE)
    with torch.no_grad():
        logits = model(xb)
        probs = F.softmax(logits, dim=1)
    return probs.cpu()

def entropy_from_probs(probs: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    p = probs.clamp(min=eps)
    return -(p * p.log()).sum(dim=1)

def ensemble_probs_probavg(models, x01_batch: torch.Tensor) -> torch.Tensor:
    probs_list = [predict_proba_one_model(m, x01_batch) for m in models]
    probs_stack = torch.stack(probs_list, dim=0)   
    return probs_stack.mean(dim=0)

def ensemble_pred_majority(models, x01_batch: torch.Tensor) -> torch.Tensor:
    preds = []
    for m in models:
        probs = predict_proba_one_model(m, x01_batch)
        preds.append(probs.argmax(dim=1))
    preds_stack = torch.stack(preds, dim=0)        
    voted, _ = torch.mode(preds_stack, dim=0)
    return voted


# In[14]:


# ImageNet normalization 
weights = ResNet18_Weights.IMAGENET1K_V1
mean, std = weights.transforms().mean, weights.transforms().std

mean_t = torch.tensor(mean).view(3, 1, 1)
std_t  = torch.tensor(std).view(3, 1, 1)

def normalize_for_model(x01: torch.Tensor) -> torch.Tensor:
    """
    x01: Tensor [3,H,W] in [0,1]
    returns ImageNet-normalized tensor
    """
    return (x01 - mean_t) / std_t

print("normalize_for_model is now defined.")


# In[15]:


angles = list(range(0, 181, 10)) 
acc_single = []
acc_maj = []
acc_prob = []

ent_single = []
ent_prob = []

for a in angles:
    xr = rotate_batch(x0, a)

    # single
    ps = predict_proba_one_model(best_model, xr)
    pred_s = ps.argmax(dim=1)
    acc_single.append((pred_s == y0).float().mean().item())
    ent_single.append(entropy_from_probs(ps).mean().item())

    # ensemble majority
    pred_m = ensemble_pred_majority(models_ens, xr)
    acc_maj.append((pred_m == y0).float().mean().item())

    # ensemble prob-avg
    pe = ensemble_probs_probavg(models_ens, xr)
    pred_p = pe.argmax(dim=1)
    acc_prob.append((pred_p == y0).float().mean().item())
    ent_prob.append(entropy_from_probs(pe).mean().item())

print("Done. Angles:", len(angles))


# In[16]:


plt.figure(figsize=(7,5))
plt.plot(angles, acc_single, marker="o", label="Best single model")
plt.plot(angles, acc_maj, marker="o", label="Ensemble majority vote")
plt.plot(angles, acc_prob, marker="o", label="Ensemble prob-avg")
plt.xlabel("Rotation angle (degrees)")
plt.ylabel("Accuracy on 20 MNIST images")
plt.title("Exp5: Accuracy vs rotation angle (0°–180°)")
plt.ylim(0, 1.05)
plt.grid(True)
plt.legend()
plt.show()


# In[17]:


plt.figure(figsize=(7,5))
plt.plot(angles, ent_single, marker="o", label="Single model entropy")
plt.plot(angles, ent_prob, marker="o", label="Ensemble prob-avg entropy")
plt.xlabel("Rotation angle (degrees)")
plt.ylabel("Mean predictive entropy (nats)")
plt.title("Exp5: Uncertainty vs rotation angle")
plt.grid(True)
plt.legend()
plt.show()


# In[18]:


example_i = 0  # choosing any index 0..19
fig, axes = plt.subplots(1, len(angles), figsize=(2*len(angles), 2))
fig.suptitle(f"Example digit (true label={y0[example_i].item()}) rotated 0°–180°", fontsize=14)

for j, a in enumerate(angles):
    img = rotate_batch(x0[example_i:example_i+1], a)[0].permute(1,2,0).clamp(0,1)
    axes[j].imshow(img)
    axes[j].axis("off")
    axes[j].set_title(f"{a}°", fontsize=10)

plt.tight_layout()
plt.show()


# ### **Rotation Robustness:**
# 
# The results show that classification accuracy on MNIST images strongly depends on the applied rotation angle. Performance remains high for small rotations but degrades rapidly beyond approximately 30°, reaching a minimum around 90°–100°, where digits are most distorted relative to the training distribution. Interestingly, accuracy partially recovers for larger rotations, reflecting the fact that some rotated digits resemble other valid digits.
# 
# If only a single model were used, the classifier lacks rotation invariance and fails significantly under geometric transformations, while providing limited insight into prediction reliability.
# 
# Using multiple models improves robustness analysis by producing smoother accuracy curves and, more importantly, by providing a stronger uncertainty signal. The ensemble exhibits higher predictive entropy in regions where accuracy collapses, indicating model disagreement and reduced confidence. Although ensembles do not solve the rotation sensitivity problem, they enable more reliable identification of unreliable predictions and offer a clearer understanding of model limitations.

# In[ ]:




