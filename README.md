# PAttn: Patch-based Attention for Long-Term Time Series Forecasting

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Topic: Time Series](https://img.shields.io/badge/Task-Time%20Series%20Forecasting-blueviolet.svg)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

This repository contains the implementation of **PAttn** (Patch-based Attention Mechanism) and the baseline **PatchTST** for multivariate and univariate long-term time series forecasting (LTSF). The project includes a complete modular training pipeline, experimental Jupyter notebooks evaluated on public benchmarks and a private cryptocurrency dataset (`BTC/USDT` 15-minute timeframe), and the presentation slide deck (`BTL-Timeseries.pptx`).

---

## 🌟 Key Features

- **Patching Segmentation (Patch-level Tokenization):** Groups consecutive time steps into sub-series patches. This retains local semantic dependencies, suppresses point-level noise, and significantly reduces the computational complexity of the Self-Attention mechanism from quadratic $O(L^2)$ to $O(N^2)$ (where $N = \lfloor (L - P) / S \rfloor + 2 \ll L$).
- **Channel Independence:** Each univariate series is processed independently through shared model backbones, mitigating cross-channel overfitting while preserving robust multi-variate modeling capacity.
- **Versatile Forecasting Modes:**
  - `M` (Multivariate $\rightarrow$ Multivariate)
  - `S` (Univariate $\rightarrow$ Univariate)
  - `MS` (Multivariate $\rightarrow$ Univariate)
- **Extensive Benchmarking:** Pre-configured for standard public time series benchmarks as well as real-world financial high-frequency data (`BTCUSDT_15m_calf.csv`).
- **Complete End-to-End Pipeline:** From data normalization and temporal embedding to training, validation with early stopping, evaluation metrics (MSE, MAE), and visualization.

---

## 🗂️ Repository Structure

```text
.
├── BTCUSDT_15m_calf.csv             # Cryptocurrency trading dataset (Bitcoin/USDT, 15m intervals)
├── BTL-Timeseries.pptx              # Presentation slides for Time Series Analysis project
├── pattn-with-dataset-public.ipynb  # Interactive notebook: experiments on public benchmark datasets
├── pattn-with-dataset-private.ipynb # Interactive notebook: experiments on private BTC/USDT dataset
└── PAttn/                           # Core framework source code
    ├── data_provider/               # Data loaders, data factory, and preprocessing pipelines
    │   └── data_factory.py
    ├── models/                      # Deep learning model architectures
    │   ├── PAttn.py                 # PAttn network architecture
    │   ├── PatchTST.py              # PatchTST baseline architecture
    │   └── Attention.py             # Multi-Head Attention and Self-Attention mechanisms
    ├── embed.py                     # Patch embedding, positional, and temporal feature encodings
    ├── main.py                      # CLI entry point for model training, evaluation, and testing
    ├── script/                      # Shell/bash scripts for batch experiment executions
    └── utils/                       # Metric calculation, EarlyStopping, and visual helpers
        └── tools.py
```

---

## 🏗️ Model Architecture & Workflow

The PAttn forecasting pipeline processes time series through the following stages:

```mermaid
flowchart LR
    A["Raw Time Series (L, M)"] --> B["Instance Normalization (RevIN)"]
    B --> C["Patching & Linear Embedding"]
    C --> D["Positional Encoding"]
    D --> E["Patch Attention Backbone"]
    E --> F["Flatten Head / Linear Projection"]
    F --> G["Instance Denormalization"]
    G --> H["Forecast Output (T, M)"]
```

1. **Instance Normalization (RevIN):** Mitigates non-stationarity and distribution shift between training and test sets.
2. **Patch Extraction & Linear Projection:** Splits lookback window $L$ into patches of length $P$ with stride $S$, then maps each patch into dimension $D$.
3. **Multi-Head Patch Attention:** Captures long-range temporal correlations across patches.
4. **Projection Head:** Flattens representations and projects to prediction horizon $T$.
5. **Denormalization:** Restores output predictions to the original data scale.

---

## 💻 Installation

### 1. Clone the repository
```bash
git clone https://github.com/TrongManh1105/Pattn.git
cd Pattn
```

### 2. Create environment & install dependencies
Python **3.8+** and PyTorch with CUDA support (optional but recommended for GPU acceleration) are recommended:

```bash
# Optional: create a virtual environment
python -m venv venv
# Linux/macOS:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Install required packages
pip install torch numpy pandas matplotlib scikit-learn tqdm
```

---

## 🚀 Usage & Training

### 1. Training via Command Line (`main.py`)

Navigate into the `PAttn/` directory and run `main.py` with the appropriate arguments:

```bash
cd PAttn

python main.py \
  --model_id btc_15m_experiment \
  --data custom \
  --root_path ../ \
  --data_path BTCUSDT_15m_calf.csv \
  --features M \
  --seq_len 512 \
  --pred_len 96 \
  --patch_size 16 \
  --kernel_size 25 \
  --batch_size 128 \
  --learning_rate 0.0001 \
  --train_epochs 20 \
  --patience 5 \
  --loss_func mse
```

#### Key Command-Line Arguments:

| Argument | Type | Default | Description |
|---|---|---|---|
| `--model_id` | str | `'test'` | Experiment identifier name |
| `--data` | str | `'custom'` | Dataset type (`custom`, `traffic`, `weather`, etc.) |
| `--root_path` | str | `'./dataset/traffic/'` | Root directory containing the data file |
| `--data_path` | str | `'traffic.csv'` | Name of the CSV dataset file |
| `--features` | str | `'M'` | Forecasting mode: `M` (Multivariate), `S` (Univariate) |
| `--seq_len` | int | `512` | Input sequence lookback window length |
| `--pred_len` | int | `96` | Future forecasting horizon length |
| `--patch_size` | int | `16` | Length of each sub-series patch |
| `--learning_rate` | float | `0.0001` | Initial optimizer learning rate |
| `--batch_size` | int | `512` | Batch size for training and testing |
| `--train_epochs` | int | `10` | Maximum number of training epochs |
| `--patience` | int | `3` | Early stopping patience epochs |
| `--loss_func` | str | `'mse'` | Loss function (`mse`, `mae`) |

---

### 2. Interactive Jupyter Notebooks

For step-by-step execution, parameter tuning, and visual plots:

- **[`pattn-with-dataset-public.ipynb`](pattn-with-dataset-public.ipynb):** Walkthrough on benchmark public datasets with data loading, feature scaling, model training, and performance evaluation.
- **[`pattn-with-dataset-private.ipynb`](pattn-with-dataset-private.ipynb):** Specialized pipeline on the Bitcoin 15-minute dataset (`BTCUSDT_15m_calf.csv`), showcasing preprocessing, training curves, and Ground Truth vs. Prediction visualization plots.

---

## 📈 Evaluation & Results

Models are evaluated using standard time series metrics:
- **Mean Squared Error (MSE):** $\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$
- **Mean Absolute Error (MAE):** $\text{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$

Detailed analysis, experimental graphs, and comparative discussions are provided in **`BTL-Timeseries.pptx`**.

---

## 📑 Project Presentation

Comprehensive theoretical foundations, empirical analysis, and comparative benchmarking can be found in the project slide deck:
- 📄 **[`BTL-Timeseries.pptx`](BTL-Timeseries.pptx)**

---

## 👤 Author

- GitHub: [@TrongManh1105](https://github.com/TrongManh1105)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
