# Out-of-Distribution Detection & Neural Collapse on CIFAR-100

This repository contains the implementation and analysis for a practical assignment of the course  
**5IA23 – Deep Learning Based Computer Vision** at **ENSTA Paris**.

The project focuses on **Out-of-Distribution (OOD) detection** and the **Neural Collapse phenomenon**, combining empirical analysis with modern OOD scoring methods on deep neural networks.

---

## 🎓 Course Information

- Course: 5IA23 – Deep Learning Based Computer Vision  
- Institution: ENSTA Paris  
- Program: Engineering cycle / MVA  
- Topic: OOD Detection & Neural Collapse

---

## 📌 Objectives

The main objectives of this assignment are:

- Train a **ResNet-18** classifier on **CIFAR-100**
- Implement and compare multiple **OOD detection scores**
- Study the **Neural Collapse phenomenon (NC1–NC5)** at the end of training
- Implement **NECO (Neural Collapse Inspired OOD Detection)**
- (Bonus) Analyze Neural Collapse across intermediate layers

---

## 🧠 Methods Implemented

### In-Distribution Training
- Dataset: CIFAR-100
- Architecture: ResNet-18
- Framework: PyTorch

### OOD Scoring Methods
- Max Softmax Probability (MSP)
- Maximum Logit Score
- Mahalanobis Distance
- Energy Score
- ViM
- NECO (Neural Collapse Inspired OOD Detection)

---

## 📊 Neural Collapse Analysis

We analyze the following Neural Collapse properties:

- NC1 – Within-class variance collapse  
- NC2 – Class means converge to a simplex Equiangular Tight Frame (ETF)  
- NC3 – Alignment between classifier weights and class means  
- NC4 – Self-duality between features and classifier weights  
- NC5 – Consequences for OOD detection

Visualizations include:
- Class mean distances
- Within-class variance
- Cosine similarity between classifier weights and class means

---

## 📈 Deliverables

- Training curves and test accuracy
- Quantitative comparison of OOD detection methods
- Neural Collapse visualizations
- Bonus: Neural Collapse behavior across layers

---

## 🗂 Repository Structure

.
├── data/                # Dataset loading and preprocessing
├── models/              # ResNet-18 and feature extractors
├── training/            # Training and evaluation scripts
├── ood/                 # OOD scoring methods
├── neural_collapse/     # Neural Collapse metrics and analysis
├── experiments/         # Experiment configurations
├── notebooks/           # Analysis and visualization notebooks
├── results/             # Plots and saved metrics
└── README.md

---

## 🚀 Usage

1. Install dependencies:
pip install -r requirements.txt

2. Train the classifier:
python training/train.py

3. Evaluate OOD detection:
python ood/evaluate_ood.py

4. Analyze Neural Collapse:
python neural_collapse/analyze_nc.py

---

## 📚 References

- Papyan et al., Neural Collapse, 2020  
- Liu et al., Energy-based Out-of-Distribution Detection, 2020  
- Wang et al., ViM: Out-of-Distribution with Virtual-logit Matching, 2022  

---

## 👤 Author

Vitor Odorissio  
ENSTA Paris
