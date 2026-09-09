# MPUP Pipeline — common workflows
# Ganti DATASET, SOURCE_MODEL, TARGET_MODEL sesuai kebutuhan

DATASET       ?= ragbench
SOURCE_MODEL  ?= meta-llama/Llama-3.2-1B-Instruct
TARGET_MODEL  ?= meta-llama/Llama-3.1-8B-Instruct
SPLIT         ?= validation
MAX_SAMPLES   ?= 500
K_SHOTS       ?= 10
CONFIG        ?= configs/default.yaml

RAW_OUT       := data/raw/$(DATASET)_labeled.jsonl
SCORED_OUT    := data/raw/$(DATASET)_scored.jsonl
FEATURES      := data/processed/$(DATASET)_features.npy
LABELS        := data/processed/$(DATASET)_labels.npy
CHECKPOINT    := checkpoints/mpup_$(DATASET)
RESULTS_DIR   := results

.PHONY: setup label score features train evaluate ablation all

## Install deps
setup:
	pip install -r requirements.txt
	python -m spacy download en_core_web_sm

## Step 1: label utility scores (u_i_true) dengan source LLM
label:
	python scripts/label_utility.py \
		--dataset $(DATASET) \
		--split $(SPLIT) \
		--source-model $(SOURCE_MODEL) \
		--output $(RAW_OUT) \
		--max-samples $(MAX_SAMPLES)

## Step 2: tambah BM25 retrieval scores ke setiap passage
score:
	python scripts/retrieve_and_score.py \
		--input $(RAW_OUT) \
		--output $(SCORED_OUT)

## Step 3: ekstrak feature vectors D1-D5
features:
	python scripts/extract_features.py \
		--config $(CONFIG) \
		--data $(SCORED_OUT) \
		--model $(TARGET_MODEL) \
		--output-dir data/processed

## Step 4: train MPUP predictor
train:
	python scripts/train.py \
		--config $(CONFIG) \
		--features $(FEATURES) \
		--labels $(LABELS) \
		--output $(CHECKPOINT)

## Step 5: evaluate vs all baselines
evaluate:
	python experiments/evaluate_baselines.py \
		--config $(CONFIG) \
		--features $(FEATURES) \
		--labels $(LABELS) \
		--checkpoint $(CHECKPOINT) \
		--dataset $(DATASET) \
		--output-dir $(RESULTS_DIR)

## Jalankan ablation study (drop satu feature group per run)
ablation:
	python - <<'EOF'
import numpy as np, joblib, yaml
from experiments.ablation import run_ablation
from src.predictor.mpup_predictor import MPUPPredictor

X = np.load("$(FEATURES)")
y = np.load("$(LABELS)")
with open("$(CONFIG)") as f:
    cfg = yaml.safe_load(f)
pred = MPUPPredictor.load("$(CHECKPOINT)", algorithm=cfg["predictor"]["algorithm"])
results = run_ablation(X, y, pred)
for k, v in results.items():
    print(f"{k:<30} {v:.4f}")
EOF

## Run full pipeline (label → score → features → train → evaluate)
all: label score features train evaluate
