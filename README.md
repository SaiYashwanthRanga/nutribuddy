# RadioML Source Classification with CNN

This repository now includes a complete, reproducible pipeline for training a
convolutional neural network (CNN) that classifies modulation sources in the
open-source [RadioML 2016.10a](https://www.deepsig.ai/datasets) dataset. The
pipeline covers dataset acquisition, preprocessing, model definition, training,
and evaluation. It also exposes a simple command-line interface so you can train
or fine-tune the model on your own hardware.

> **Note**
> The original NutriBuddy documentation has been retained at the end of this
> file for archival purposes.

## Project layout

```
├── pyproject.toml            # Python package definition and dependencies
├── scripts
│   └── train_cnn.py          # CLI entry point for model training
├── src
│   └── rml_cnn
│       ├── __init__.py       # Package exports
│       ├── data.py           # Dataset download and preparation utilities
│       ├── model.py          # CNN architecture
│       └── train.py          # Training loop helpers
└── tests                     # Unit tests covering the data/model pipeline
```

## Getting started

1. **Create and activate a virtual environment** (recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. **Install the project in editable mode with the development extras**:
   ```bash
   pip install -e .[dev]
   ```

3. **Install PyTorch (optional runtime dependency)**. The training script uses
   PyTorch; install the CPU build that matches your platform. For example:
   ```bash
   pip install 'rml-cnn[torch]'
   ```
   If you prefer a specific wheel (e.g., from a vendor mirror), install it
   directly before running the command above.

4. **Download the RadioML 2016.10a dataset**. The project ships with a helper
   that knows about several public mirrors. Pick one that is convenient for you:

   ```bash
   python scripts/train_cnn.py --download --data-dir data \
       --dataset-url https://huggingface.co/datasets/RadioML/RadioML2016.10a/resolve/main/RML2016.10a_dict.pkl \
       --max-samples 20000 --epochs 5 --batch-size 512
   ```

   The `--max-samples` flag lets you work with a manageable subset while you are
   experimenting. Remove the flag to train on the full dataset. Training logs and
   metrics are written to the `artifacts/` directory by default.

5. **Review metrics** stored in `artifacts/metrics.json` and the saved model in
   `artifacts/rml_cnn.pt`. The JSON file contains epoch-by-epoch loss and
   accuracy along with held-out test performance.

### Programmatic usage

You can also use the package directly from Python:

```python
from pathlib import Path

from rml_cnn import (
    RMLCNN,
    download_rml2016_dataset,
    load_rml2016_dataset,
    prepare_rml2016_datasets,
    train_model,
)

dataset_path = download_rml2016_dataset(Path("data"))
raw = load_rml2016_dataset(dataset_path)
splits, encoder = prepare_rml2016_datasets(raw, max_samples=20000)
model = RMLCNN(input_channels=splits["train"][0].shape[1], num_classes=len(encoder.classes_))
model, history, test_metrics = train_model(model, splits, epochs=5)
print(test_metrics)
```

## Running the tests

The repository includes lightweight tests that validate data preparation and the
model’s forward pass. After installing the development dependencies, run:

```bash
pytest
```

## Dataset mirrors

The downloader attempts the following mirrors by default:

1. `https://huggingface.co/datasets/RadioML/RadioML2016.10a/resolve/main/RML2016.10a_dict.pkl`
2. `https://zenodo.org/record/2680980/files/RML2016.10a_dict.pkl?download=1`
3. `https://storage.googleapis.com/radio_ml_data_public/RML2016.10a_dict.pkl`

If one mirror is temporarily unavailable, the script will automatically fall
back to the next URL.

---

## Legacy: NutriBuddy – A Diet Plan Recommendation System

*(Content preserved from the original repository description.)*

NutriBuddy is an AI-powered chatbot designed to predict chronic diseases based
on symptoms and provide personalized diet recommendations. The system leverages
Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) to enhance
accuracy and provide relevant dietary suggestions. The chatbot can predict
illnesses such as diabetes, heart disease, and thyroid disorders based on
symptoms and suggest suitable diet plans accordingly.

### Key Features

- **Disease Prediction** – Users can enter symptoms, and the chatbot will
  predict possible diseases using LLM-based reasoning and vector embeddings.
- **Diet Plan Recommendations** – Provides personalized diet plans based on the
  detected disease with nutrient-rich food options and items to avoid.
- **Conversational AI** – Interactive chatbot interface for real-time user
  engagement with context-aware follow-up questions.
- **Data-Driven Insights** – Uses a vector database (Chroma DB) and PostgreSQL
  for storing medical and nutritional data.

*(See the original README content above for full installation and usage
details.)*
