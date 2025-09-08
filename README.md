# StoDir - Stock Direction Forecasting Application

An end-to-end system for training, evaluating, and deploying a stock-direction forecasting model.

![Last Commit](https://img.shields.io/github/last-commit/Asifdotexe/StoDir)
![Top Language](https://img.shields.io/github/languages/top/Asifdotexe/StoDir)
![Languages Count](https://img.shields.io/github/languages/count/Asifdotexe/StoDir)
[![Deployed App](https://img.shields.io/badge/Deployed%20App-Live-green)](https://stodirforecast.streamlit.app/)

---

## Table of Contents

- [Overview](#overview)
  - [What is StoDir?](#what-is-stodir)
  - [Key technical features](#key-technical-features)
  - [How it works](#how-it-works)
  - [Tech stack](#tech-stack)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Usage](#usage)
- [Documentation](#documentations)
- [License](#license)

---

## Overview

### What is StoDir?
StoDir is a machine learning system that forecasts the likely direction (Up or Down) of a stock's price for the next trading day. This project is a complete, well-architected system that demonstrates professional techniques for building and deploying AI models.

> Disclaimer: This is a portfolio project designed to showcase technical skills and is not financial advice.

### Key Technical Features
1. A clear separation between the model training pipeline and the live inference application. The model is a versioned artifact, loaded by the app for fast, on-demand predictions.
2. Employed a sliding-window backtesting methodology to provide a realistic performance assessment, mitigating lookahead bias common in financial forecasting.
3. Utilized Hugging Face Hub as a model registry to store and version the trained model artifacts, enabling reproducible deployments.
4. Provides both a Streamlit web interface for interactive use and a command-line interface (CLI) for programmatic access.

### How it works
The system is split into two parts: an offline Training Pipeline that builds the model and a live Application that uses the model to make predictions.

```text

                            +--------------------------+
                            |   Hugging Face Hub       |
(1) TRAINING (Offline)      |  (Cloud Model Storage)   |     (3) PREDICTION (Live)
+----------------------+    | +--------------------+   |    +----------------------+
| train.py             |    | | stodir_model.joblib|   |    | app.py (Web App)     |
| (Uses historical     |--->| +--------------------+   |--->| (Downloads model     |
| data to build model) |    +--------------------------+    | & makes predictions) |
+----------------------+                                    +----------------------+


```

For a more detailed explanation, see the [Architecture Document](docs/SYSTEM_ARCHITECTURE.md).

### Tech Stack
- ML & Data: Scikit-learn, Pandas, yfinance
- Deployment & MLOps: Hugging Face Hub, Streamlit, Joblib
- Tooling: PyYAML, Pytest, Pylint

---

## Getting Started

### Prerequisites

1. Using the Deployed Web App
The easiest way to use the application is to visit the live version hosted on Streamlit Cloud:

> Visit: [StoDir on Streamlit](https://stodirforecast.streamlit.app/)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Asifdotexe/StoDir.git
   cd StoDir
2. It is recommended to use a virtual environment. Here are two common ways to set it up:

    <details> <summary>Using <strong>venv</strong></summary>
    
        # Create virtual environment
        python -m venv venv

        # Activate on Windows
        venv\Scripts\activate

        # Activate on macOS/Linux
        source venv/bin/activate

    </details> <details> <summary>Using <strong>conda</strong></summary>

        # Create new conda environment
        conda create -n stodir-env python=3.12

        # Activate the environment
        conda activate stodir-env

    </details>

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Usage

1. **Run the Web App**

    Starting a local web server. The app uses the pre-trained model from Hugging Face.

    ```bash
    streamlit run app.py
    ```
    If your Hub repo is private, set `HF_TOKEN` in the environment before starting.

2. **Use the Command-Line Tool**

    The CLI uses a local `artifacts/stodir_model.joblib` by default. Create it by running the training pipeline (or point the CLI to a Hub-hosted model).

    ```bash
    python train.py
    ```
    Then, get a forecast directly in your terminal:

    ```bash
    python cli.py GOOGL
    ```

## Documentation
- [Project Methodology Document](docs/PROJECT_METHODOLOGY.md)
- [System Architecture Document](docs/SYSTEM_ARCHITECTURE.md)
- [Backtesting Document](docs/BACKTESTING.md)

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.