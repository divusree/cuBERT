# NER Pre-training with Vanilla BERT

This project implements a Named Entity Recognition (NER) task using a custom BERT model with a vanilla encoder stack.

## Features

- Custom BERT model with token classification head for NER.
- Preprocessing pipeline for tokenization and label alignment.
- Training with gradient accumulation, learning rate scheduling, and TensorBoard logging.
- Evaluation using `seqeval` for F1-score and classification reports.

## Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset

Uses the CoNLL-2003 dataset, automatically downloaded via the `datasets` library.

## Training

1. Open `train.ipynb`.
2. Run cells to:
   - Load data and initialize the model.
   - Train and evaluate the model.
3. Save the trained model:
   ```python
   torch.save(model.state_dict(), 'final.pt')
   ```

## Key Files

- **`BERT.py`**: Implements the BERT model and NER head.
- **`BERTdataloader.py`**: Handles data preprocessing and batching.
- **`train.ipynb`**: Training and evaluation script.

## Results

Metrics (e.g., F1-score) are logged in TensorBoard and printed during training.
