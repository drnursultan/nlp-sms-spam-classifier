# Quick helpers
.PHONY: train inspect roc cm tune clean

train:
	python src/train.py --data data/raw/SMSSpamCollection

inspect:
	python src/top_ngrams.py --model_dir models/latest --k 25

roc cm:
	python - <<'PY'
import nbformat, nbconvert, sys
print('Open and run notebooks/02_eval_plots.ipynb in Jupyter to generate plots.')
PY

tune:
	python - <<'PY'
import nbformat, nbconvert, sys
print('Open and run notebooks/03_threshold_tuning.ipynb in Jupyter to sweep thresholds.')
PY

clean:
	rm -rf models/latest models/run-* reports/top_features_*.txt reports/figures/*.png
