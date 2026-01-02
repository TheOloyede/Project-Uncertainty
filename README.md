# Project Uncertainty

## Exploring Predictive Uncertainty and Robustness with Deep Ensembles

---

### Project Overview
This project investigates **predictive uncertainty in deep learning models** using **deep ensembles**, following the ideas introduced in:
*Lakshminarayanan et al., “Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles”*

A **ResNet18 classifier** trained on the **MNIST dataset** is used to analyze how predictions behave under:
* Additive noise
* Out-of-distribution data
* Geometric transformations (rotation)

The project also studies how ensembles of models improve robustness and uncertainty estimation compared to a single neural network. The work is organized into **five experiments**, each focusing on a different robustness or uncertainty aspect.

---

### Repository Structure

```text
.
├── Experiment_1.py # Training ResNet18 (scratch vs pretrained)
├── Experiment_2.py # Robustness to white noise (single model)
├── Experiment_3.py # Noise robustness: single vs ensemble (7 models)
├── Experiment_4.py # Out-of-distribution test (MNIST vs KMNIST)
├── Experiment_5.py # Rotation robustness (0°–180°)
├── outputs/
│ ├── models/ # Saved trained models (.pt)
│ ├── tensorboard/ # Training logs
│ └── figures/ # Generated plots
└── README.md
```

### Requirements
* Python ≥ 3.9
* PyTorch
* torchvision
* numpy
* matplotlib

**Installation:**
```bash
pip install torch torchvision numpy matplotlib
```

### Experiments Summary

#### **Experiment 1 — Pretraining vs Training from Scratch**
A ResNet18 model is trained on MNIST once from scratch and once using ImageNet pretraining. The evolution of loss and convergence speed are analyzed.
> **Key Observation:** Pretrained networks converge faster and exhibit more stable training behavior, especially during early epochs.

#### **Experiment 2 — Robustness to White Noise (Single Model)**
Evaluates a trained ResNet18 model on 20 MNIST test images while increasing levels of white noise are added to the input.

> **Key Observation:** A significant drop in classification accuracy occurs before strong visual degradation, indicating neural networks are more sensitive to noise than human perception.

#### **Experiment 3 — Ensemble Robustness Under Noise**
Seven independent ResNet18 models are trained using different random seeds. Performance is compared between the best single model, ensemble majority voting, and ensemble probability averaging.
> **Key Observations:**
> * **Variance Reduction:** Ensembles reduce variance and stabilize predictions.
> * **Smoother Curves:** Probability averaging produces smoother degradation curves.
> * **Reliability:** Ensembles improve reliability rather than guaranteeing higher peak accuracy.

#### **Experiment 4 — Out-of-Distribution Detection (MNIST vs KMNIST)**
Models trained on MNIST are evaluated on MNIST test images (in-distribution) and KMNIST images (out-of-distribution).

> **Key Observations:**
> * **Confidently Wrong:** Single models are "confidently wrong" on KMNIST.
> * **Entropy Shift:** Ensemble entropy increases significantly on out-of-distribution data.

#### **Experiment 5 — Rotation Robustness**
MNIST test images are rotated from 0° to 180°, and classification accuracy and uncertainty are measured.

> **Key Observations:**
> * **Accuracy Collapse:** Accuracy remains high for small rotations and collapses around 90°–100°.
> * **Entropy Peak:** Predictive entropy peaks where accuracy is lowest.
> * **Interpretability:** Ensemble results are smoother and more interpretable than single-model results.

---

### Uncertainty Quantification
Predictive uncertainty is quantified using **Shannon Entropy**:

**H(p) = − Σ p(c) log p(c)**

**Higher entropy corresponds to:**
* Lower confidence
* Increased disagreement between models

---

### Main Conclusions
* **Sensitivity:** Deep neural networks are highly sensitive to noise, rotation, and distribution shifts.
* **Vulnerability:** Severe performance degradation can occur without obvious visual corruption.
* **Ensembles:** Reduce prediction variance, improve robustness analysis, and provide meaningful uncertainty estimates.
* **Reliability:** Uncertainty estimation is essential for **reliable and safe decision-making**, especially under dataset shift.

---

### References
* [A Comprehensive Introduction to Uncertainty in Machine Learning](https://imerit.net/resources/blog/a-comprehensive-introduction-to-uncertainty-in-machine-learning-all-una/)
* [Lakshminarayanan et al., Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles](https://doi.org/10.48550/arXiv.1612.01474)
* [Rotation Robustness Reference](https://doi.org/10.48550/arXiv.1806.01768)
