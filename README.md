
# Heart Disease Prediction — Machine Learning Mini Project

## Live Link
https://tool-heart-disease-prediction.streamlit.app/

## 1. Project Overview

This project demonstrates an end-to-end machine-learning workflow
using the UCI Heart Disease dataset.

The application accepts dataset features and displays the
classification produced by a trained machine-learning model.

This is an educational prototype, not a medical diagnostic tool.

## 2. Dataset

Source: UCI Machine Learning Repository

Dataset: Heart Disease (ID 45)

URL: https://archive.ics.uci.edu/dataset/45/heart+disease

The project uses the Cleveland subset.

The original target `num` has values 0 through 4.
For binary classification:
- 0 = negative class
- 1, 2, 3, 4 = positive class

The target transformation and missing-value handling are
implemented in `train_model.py`.

## 3. Technologies

- Python
- Pandas and NumPy
- Scikit-learn
- Jupyter Notebook
- Streamlit
- Joblib
- Git and GitHub

## 4. Project Structure

```text
heart-disease-ml/
├── data/
│   └── cleveland.csv
├── notebooks/
│   └── analysis.ipynb
├── train_model.py
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## 5. Installation

Create and activate a virtual environment, then install dependencies.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate the environment using:

```bash
.venv\Scripts\activate
```

## 6. Train and Evaluate

Place `cleveland.csv` inside the `data/` directory.

Run:

```bash
python train_model.py
```

The script compares Logistic Regression, Decision Tree,
and Random Forest using five-fold cross-validation and
hyperparameter tuning.

It evaluates the selected model on a held-out test set
and saves `model.pkl` and `model_comparison.csv`.

## 7. Run the Streamlit Application

After training:

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal.

## 8. Evaluation

Report the measured:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC, where supported
- Confusion matrix

Actual scores must be taken from the executed training script.

## 9. Limitations

The dataset is small and historical. Model performance on
this dataset does not establish clinical validity or
reliability on modern patient populations.

The application is intended for educational demonstration only.

## 10. Dataset Citation

Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989).
Heart Disease. UCI Machine Learning Repository.
https://doi.org/10.24432/C52P4X

Dataset license: CC BY 4.0.
