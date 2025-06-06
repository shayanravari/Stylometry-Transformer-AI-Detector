# AI Text Detection System - Final Project

This project is a system for detecting AI-generated text, developed for the CS 162 final project. It includes a Naive Bayes baseline and a fine-tuned BERT-based classifier.

## Team Members
- Shayan Ravari
- Rohith Venkatesh
- Rishi Padmanabhan

## Setup

1.  Clone the repository.
2.  Create a Python virtual environment (e.g., `python -m venv venv` and `source venv/bin/activate`).
3.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
4.  Ensure the training data (`real-vs-gpt2-sentences.jsonl`, `HC3.jsonl`) is in the `./data/` directory and the development set is in the `./devset/` directory.

## How to Run

The project is structured into three main steps, executed via Jupyter notebooks or Python scripts.

**Step 1: Train the Naive Bayes Baseline Model:** 
Run the `baseline_classifier.ipynb` notebook. This will train the model and save the classifier and TF-IDF vectorizer to the `./baseline_saved_model/` directory.
```bash
jupyter notebook baseline_classifier.ipynb
```

**Step 2: Train the BERT Model:**
Run the `train_bert.ipynb` notebook. This will fine-tune a bert-base-uncased model using the combined training data and save the best-performing model to the `./bert_ai_detector_final/` directory. Training may take some time and requires a GPU for reasonable performance.
```bash
jupyter notebook train_bert.ipynb
```

**Step 3: Evaluate Models on the Development Set:**
Run the `evaluate_on_dev.ipynb` notebook. This script loads both the saved Naive Bayes and BERT models and evaluates them against the provided dev set. It will print a summary of performance metrics to the console and save detailed evaluation plots (confusion matrices, ROC curves, etc.) to the `./evaluation_outputs/` directory.
```bash
jupyter notebook evaluate_on_dev.ipynb
```
