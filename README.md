# 💾 Machine Learning Model Persistence Demo

This repository contains a simple Python script (`model_persistence_demo.py`) that demonstrates the complete workflow of training a basic Machine Learning model, saving it to disk, loading it back, and then using the loaded model to make predictions. This is a crucial step for deploying ML models in real-world applications.

## 🌟 Core Concept Demonstrated

* **Model Persistence:** The process of saving a trained Machine Learning model to a file and loading it later, eliminating the need for retraining.
* **Scikit-learn:** Basic usage of `LinearRegression` for supervised learning.
* **`pickle` module:** Python's standard library for object serialization (saving/loading Python objects).

## 🚀 Getting Started

Follow these steps to set up and run the demo.

### Prerequisites

* Python 3.8+ (Recommended)
* `pip` (Python package installer)

### Installation

1.  **Clone the repository (or create the file directly):**
    If you've already cloned your main learning repository, navigate to the directory where `model_persistence_demo.py` is located. Otherwise, create a new folder and the `model_persistence_demo.py` file within it.

2.  **Create and activate a virtual environment** (highly recommended):
    ```bash
    python -m venv venv_model_demo
    ```
    * **On Windows (Command Prompt):**
        ```bash
        .\venv_model_demo\Scripts\activate
        ```
    * **On Windows (PowerShell):**
        ```powershell
        .\venv_model_demo\Scripts\Activate.ps1
        ```
    * **On macOS/Linux:**
        ```bash
        source venv_model_demo/bin/activate
        ```

3.  **Install project dependencies:**
    ```bash
    pip install pandas scikit-learn
    ```

### Files in this Demo

* `model_persistence_demo.py`: The main script that performs model training, saving, loading, and prediction.
* `student_scores.csv`: (Generated automatically by `model_persistence_demo.py` if not present). This is a small dummy dataset used for training the linear regression model.
* `linear_regression_model_single_file.pkl`: This file is **generated** when you run the script. It's the serialized (saved) version of your trained Linear Regression model.

## 📦 Usage

To run the demonstration:

1.  Ensure your `venv_model_demo` virtual environment is activated.
2.  Execute the Python script:
    ```bash
    python model_persistence_demo.py
    ```

### Expected Output

The script will print various messages indicating the progress:
* Data preparation details.
* Model training confirmation and coefficients.
* Messages confirming the model is being saved to `linear_regression_model_single_file.pkl`.
* Messages confirming the model is being loaded from the `.pkl` file.
* The prediction made by the **loaded model** for a new data point (6 hours studied).

You will also find a new file named `linear_regression_model_single_file.pkl` created in the same directory as your script. You can optionally uncomment the cleanup lines in the script to remove this file after the demo.

---

## 💡 Learning Context

This demo is part of a broader journey to learn Python from basics to advanced topics, with a recent focus on Data Analysis and Machine Learning fundamentals. It specifically highlights how to make trained ML models reusable without retraining.

---

## 📄 License

This project is open-source. Please refer to the main repository's `LICENSE` file for details.

---
