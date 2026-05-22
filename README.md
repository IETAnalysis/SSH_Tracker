# SSH_Tracker

Encrypted tunneling technologies provide privacy protection while concealing fine-grained behavioral semantics. When SSH traffic is transmitted through encrypted tunnels, multiple behaviors are often executed sequentially, producing implicit transitions that cannot be directly observed. **SSH-Tracker** is a multi-task learning framework designed to jointly perform behavioral boundary localization and segment-level behavior classification within a shared representation space, effectively modeling sequential SSH behaviors with unknown transition boundaries.
<img width="2365" height="1006" alt="image" src="https://github.com/user-attachments/assets/dd4d5b54-d852-4db1-800e-40db22c8d6e7" />

📊 Dataset
The dataset used in this research contains tunnel traffic constructed from six different tunneling technologies, encompassing various sequential SSH behaviors with implicit transitions.
You can download the dataset from the following Google Drive link: [Download Link](https://drive.google.com/drive/folders/1w47_5o_xvA-gDinlIw7pJ90ElpL7oIuy?usp=sharing)

## 🔍 Overview
SSH-Tracker implements a cascaded architecture with the following core mechanisms:
1. **Hierarchical Encoder**: Employs an attention residual hierarchical GRU with adaptive gating to capture both long-range temporal dependencies and local transition patterns.
2. **Multi-scale Window Statistics**: Identifies behavioral transitions by modeling contextual differences between adjacent temporal regions across multiple window sizes.
3. **Multi-task Learning**: Jointly optimizes packet-level boundary detection (Phase 1) and segment-level behavior classification (Phase 2) in an end-to-end manner.

---


## 📂 Project Structure & File Description
```text
├── models.py          # Core neural network architectures
├── train.py           # Joint training loop and loss functions
├── evaluate.py        # Evaluation metrics and inference logic
└── README.md          # Project documentation
```


## 🛠️ Installation

- Python 3.8+
- PyTorch 1.10+
- **CUDA Toolkit** (Recommended)

```bash
pip install torch torchvision torchaudio
pip install numpy tqdm tensorboard scikit-learn
```

