# SSH_Tracker

Encrypted tunneling technologies provide privacy protection while concealing fine-grained behavioral semantics. When SSH traffic is transmitted through encrypted tunnels, multiple behaviors are often executed sequentially, producing implicit transitions that cannot be directly observed. **SSH-Tracker** is a multi-task learning framework designed to jointly perform behavioral boundary localization and segment-level behavior classification within a shared representation space, effectively modeling sequential SSH behaviors with unknown transition boundaries.
<img width="2365" height="1006" alt="image" src="https://github.com/user-attachments/assets/dd4d5b54-d852-4db1-800e-40db22c8d6e7" />

📊 Dataset
The dataset used in this research contains tunnel traffic constructed from six different tunneling technologies, encompassing various sequential SSH behaviors with implicit transitions.
You can download the dataset from the following Google Drive link: [Download Link](https://drive.google.com/drive/folders/1w47_5o_xvA-gDinlIw7pJ90ElpL7oIuy?usp=sharing)

## Dataset Flow Distribution

This table reports the distribution of complete encrypted tunnel traffic flows across behavioral composition levels and tunneling technologies.
Each traffic flow corresponds to one SSH session after connection establishment. Training and test subsets are split at the complete-flow level without session overlap.

| Tunneling technology | 2-Behavior train | 2-Behavior test | 3-Behavior train | 3-Behavior test | 4-Behavior train | 4-Behavior test | 5-Behavior train | 5-Behavior test | Overall train | Overall test | Total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| VMess | 1006 | 211 | 1056 | 219 | 1118 | 218 | 1148 | 217 | 4328 | 865 | 5193 |
| TUIC | 1176 | 210 | 1169 | 203 | 1112 | 200 | 1186 | 201 | 4643 | 814 | 5457 |
| Trojan | 1169 | 210 | 1007 | 213 | 1069 | 202 | 1077 | 219 | 4322 | 844 | 5166 |
| Hysteria2 | 1174 | 211 | 1032 | 202 | 1059 | 219 | 1064 | 204 | 4329 | 836 | 5165 |
| HTTP | 1012 | 201 | 1008 | 216 | 1025 | 201 | 1139 | 216 | 4184 | 834 | 5018 |
| AnyTLS | 1045 | 205 | 1034 | 209 | 1076 | 201 | 1168 | 219 | 4323 | 834 | 5157 |
| **Total** | **6582** | **1248** | **6306** | **1262** | **6459** | **1241** | **6782** | **1276** | **26129** | **5027** | **31156** |

## Per-class Behavior-Segment Distribution

This table reports the realized class distribution in the mixed-traffic SSH dataset.
Each count denotes one labeled behavior segment, rather than a packet or a traffic flow.
The counts aggregate the 2-, 3-, 4-, and 5-behavior settings across six tunneling technologies.
For every record, the ground-truth `flow_label` sequence was verified against the contiguous segments encoded by `packet_label`.

| ID | Functional category | Behavior | Train segments | Test segments | Total segments | Share of total |
|---:|---|---|---:|---:|---:|---:|
| 0 | Administrative Querying | System Status | 9242 | 1756 | 10998 | 10.05% |
| 1 | Administrative Querying | Network Topology | 9237 | 1842 | 11079 | 10.12% |
| 2 | Administrative Querying | File System Enumeration | 9201 | 1764 | 10965 | 10.02% |
| 3 | Administrative Querying | Configuration Access | 9206 | 1773 | 10979 | 10.03% |
| 4 | Data Manipulation | Data Exfiltration | 9179 | 1754 | 10933 | 9.99% |
| 5 | Data Manipulation | Tool Implantation | 9143 | 1732 | 10875 | 9.94% |
| 6 | Data Manipulation | Bulk Output | 9142 | 1744 | 10886 | 9.95% |
| 7 | Persistent Monitoring | Resource Monitoring | 9156 | 1770 | 10926 | 9.98% |
| 8 | Persistent Monitoring | Log Auditing | 9230 | 1772 | 11002 | 10.05% |
| 9 | Persistent Monitoring | Interactive Session | 9092 | 1719 | 10811 | 9.88% |
|  | **All behaviors** |  | **91828** | **17626** | **109454** | **100.00%** |

The table covers 26,129 training flows and 5,027 test flows, for 31,156 flows in total.

## Dataset Description

### Overview

This dataset was constructed for fine-grained SSH behavior analysis in encrypted tunnel traffic. It contains continuous encrypted traffic sessions collected under six tunneling technologies: VMess, TUIC, Trojan, Hysteria2, HTTP, and AnyTLS. The dataset supports the joint evaluation of sequential SSH behavior recognition and packet-level transition-boundary localization.

Each sample represents one continuous SSH session transmitted through an encrypted tunnel. A sample contains multiple consecutive SSH behaviors rather than a single flow-level activity label. The SSH connection-establishment stage is excluded from the analyzed traffic sequence so that the data represent the subsequent behavioral execution stage.

### SSH Behavior Taxonomy

The dataset defines ten fine-grained SSH behavior categories: System Status, Network Topology, File System Enumeration, Configuration Access, Data Exfiltration, Tool Implantation, Bulk Output, Resource Monitoring, Log Auditing, and Interactive Session. These categories cover lightweight administrative querying, sustained data operations, and long-duration monitoring or interactive activities.

Each session is constructed with a predefined number of behaviors, ranging from two to five. For a given behavior-count setting, distinct behavior categories are sampled without replacement from the ten defined categories, and their execution order is randomized. This procedure avoids repeated behavior labels within an individual session while producing diverse ordered behavior compositions across the dataset.

### Collection and Annotation Protocol

For each tunneling technology, SSH traffic was collected after an encrypted tunnel had been established. Every sample corresponds to one independently collected, persistent SSH session. The behavior implementations use common SSH commands associated with the defined categories. Their durations are determined by actual command execution rather than an artificially fixed duration. Consecutive behaviors are separated by randomized waiting periods. These intervals may contain system-generated packets and tunnel-related background traffic, which are retained in the captured trace to preserve realistic temporal variation.

Behavior transitions are recorded by the collection automation script and verified using command-output records and system echo responses. The released annotations provide packet-level behavior labels. 

### Data Split and Independence

Training and testing data are split at the complete SSH-session level, with an approximately 8:2 ratio. The split is performed independently within each tunneling technology and each two- to five-behavior setting. No SSH session is duplicated across the training and testing subsets, and no continuous session is divided into multiple samples.

The repository provides the distribution of sessions across tunneling technologies, behavior-count settings, and data splits. The behavior categories are selected through the randomized sampling procedure described above.

### Session Duration and Packet Counts

Session duration is measured over the analyzed packet sequence as the cumulative inter-arrival time of its packets. Packet count is the number of packets in the same analyzed sequence. Both quantities vary across sessions because SSH commands, behavior compositions, randomized waiting periods, and tunneling technologies produce heterogeneous traffic durations and packet volumes. Consequently, the number of behaviors in a session indicates its semantic composition but does not uniquely determine its duration or packet count.



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

