# StoDir - Stock Direction Forecasting Application

*Forecast Market Moves with Confidence and Clarity*

![Last Commit](https://img.shields.io/github/last-commit/Asifdotexe/StoDir)
![Top Language](https://img.shields.io/github/languages/top/Asifdotexe/StoDir)
![Languages Count](https://img.shields.io/github/languages/count/Asifdotexe/StoDir)
[![Deployed App](https://img.shields.io/badge/Deployed%20App-Live-green)](https://stodirforecast.streamlit.app/)

---

## Table of Contents

- [Overview](#overview)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Usage](#usage)

---

## Overview

StoDir is a simple web application that forecasts the likely market direction of a given stock based on historical data. It helps users get a rough sense of how a stock might move, but it is *not* a trading strategy or advice tool — the stock market is influenced by many more factors than just past trends.

---

## Getting Started

### Prerequisites

- To use StoDir, you can simply visit the deployed [web app](https://stodirforecast.streamlit.app/).
- If you want to contribute or run it locally, clone or fork this repository and follow the setup steps below.

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
Once installed, run the app using Streamlit:
```bash
streamlit run app.py
```
This will launch a local web server. Open the URL provided in the terminal (typically http://localhost:8501) in your browser. Enter a stock ticker in the input form to see the forecasted market direction based on historical data.
