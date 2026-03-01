# Hardware-Aware Deep Learning Chess Engine

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-Int8_Quantization-005CED.svg)](https://onnxruntime.ai/)
[![License](https://img.shields.io/badge/License-EPL%202.0-red.svg)](LICENSE)

An advanced, highly optimized deep learning chess engine deployed as a live Lichess Bot. This project bridges high-level Artificial Intelligence with low-level system efficiency, focusing on **Hardware-Aware AI** and maximum inference throughput in resource-constrained (CPU-only) environments.

## Project Overview

Traditional chess engines rely heavily on deep alpha-beta pruning and hand-crafted evaluation functions. This engine replaces traditional heuristics with a deep convolutional neural network (ResNet-20) evaluated via a customized Monte Carlo Tree Search (MCTS). 

To ensure competitive latency during live gameplay, the model undergoes strict **Post-Training Quantization (PTQ)**, reducing the memory footprint and accelerating inference via the ONNX Runtime CPU Execution Provider.

### Key Technical Achievements
* **Large-Scale Distributed Training:** The ResNet-20 model was trained on **40 million positions** sampled from the Lichess Elite dataset. Evaluations were generated using Stockfish 17. Training was accelerated using `torch.nn.DataParallel` across dual NVIDIA T4 GPUs.
* **Int8 Quantization:** Converted the baseline Float32 PyTorch model to an 8-bit integer (Int8) ONNX graph, minimizing cache misses and maximizing CPU vectorization capabilities.
* **Supervised Batch MCTS:** Implemented a batched MCTS algorithm for highly parallelized tree exploration, executing ~400 simulations per move within strict time controls.

---

## System Architecture

### 1. Neural Network (ResNet-20)
The core evaluator is a 20-block Residual Network. The architecture maps an $18 \times 8 \times 8$ bitboard representation of the game state to two distinct heads:
* **Policy Head:** Outputs a probability distribution over 4,096 possible move indices.
* **Value Head:** Outputs the expected Win/Draw/Loss (WDL) continuous scalar for the current board state.

### 2. Search Algorithm (MCTS)
The search phase utilizes the PUCT (Predictor + Upper Confidence Bound applied to Trees) algorithm. During the selection phase, the engine traverses the tree by selecting nodes that maximize:

$a_t = \arg\max_a \left( Q(s, a) + C \cdot P(s, a) \cdot \frac{\sqrt{N(s)}}{1 + N(s, a)} \right)$

Where $Q(s, a)$ is the mean action value, $P(s, a)$ is the prior probability from the Policy Head, $N$ denotes visit counts, and $C$ is the exploration constant.

### 3. Inference Pipeline (ONNX)
By detaching the inference pipeline from the PyTorch backend, the engine avoids Python Global Interpreter Lock (GIL) bottlenecks during batch prediction. The dynamic quantization maps the Float32 activations and weights to `QUInt8`, achieving a ~4x reduction in model size and significantly lower latency.

---

## 📂 Repository Structure

```text
.
├── src/
│   ├── model/
│   │   ├── resnet.py          # ResNet-20 architecture definition
│   │   └── export.py          # PyTorch to ONNX Int8 quantization pipeline
│   ├── search/
│   │   └── mcts.py            # Batched MCTS and ONNX Inference Engine
│   ├── env/
│   │   └── lichess_client.py  # Asynchronous Lichess API integration
│   └── utils/
│       └── data_loader.py     # Fen-to-Tensor bitboard encoding
├── scripts/
│   └── train.py               # Multi-GPU DataParallel training loop
├── weights/                   # (Ignored) Model checkpoints and ONNX files
├── run_bot.py                 # Main entry point for the live bot
├── requirements.txt           # Dependency specifications
└── .env.example               # Environment variables template
```
