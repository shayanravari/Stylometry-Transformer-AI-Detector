# AI Text Detection System

## Setup

1.  Clone the repository.
2.  Create a Python virtual environment (e.g., `python -m venv venv` and `source venv/bin/activate`).
3.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
4.  Ensure the training data (`real-vs-gpt2-sentences.jsonl`, `HC3.jsonl`) is in the `./data/` directory and the development set is in the `./devset/` directory.

## Run Pre-trained Model

Download the `model.safetensors` file at https://ucla.box.com/shared/static/uuhwuxyacehw0mm7kymf7hvxquw8ms5r.safetensors and move it into the `bert_ai_detector_final` directory. Then, you may directly run the `evaluate_on_devset.ipynb` notebook without needing to train a new model. If you would like to test on a new development set, add your `.jsonl` file to the `devset` directory and add the name of the file to the `dev_filenames` list object in the `evaluate_on_devset.ipynb` notebook. Then, running the notebook should have it evaluate all three of the models on the additional dataset. Our main model is the hybrid model so you should pay attention to the metric scores that this model recieves.

## How to Run (General)

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

