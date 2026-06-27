.PHONY: help install install-all test lint data gen-synthetic train-baseline train evaluate demo serve autopilot grade report slides clean

help:
	@echo "mmqa — Multimodal Question Answering (VQA)"
	@echo "  install        core install (pip install -e .)"
	@echo "  install-all    full install (pip install -e .[all])"
	@echo "  test           run the test suite (CPU-only, offline)"
	@echo "  data           prefetch/sanity-check datasets (streaming probes)"
	@echo "  gen-synthetic  render synthetic scene eval images"
	@echo "  train-baseline persist the prior baseline"
	@echo "  train          fine-tune the ViLT VQA core (needs GPU)"
	@echo "  evaluate       VQA accuracy + per-type + agent coverage"
	@echo "  demo           run the agent on the synthetic seed scenes"
	@echo "  serve          start FastAPI + Gradio (/ui)"
	@echo "  autopilot      one-button train->eval->analysis->report+slides+grade"
	@echo "  grade          rubric completeness self-check"

install:
	pip install -e .

install-all:
	pip install -e .[all]

test:
	pytest -q

lint:
	ruff check src tests || true

data:
	mmqa data

gen-synthetic:
	mmqa gen-synthetic

train-baseline:
	mmqa train-baseline

train:
	mmqa train-vqa

evaluate:
	mmqa evaluate --fast

demo:
	mmqa demo-agent --fast

serve:
	mmqa serve --ui

autopilot:
	mmqa autopilot --no-train

grade:
	mmqa grade

report:
	mmqa generate-report

slides:
	mmqa generate-slides

clean:
	rm -rf build dist *.egg-info .pytest_cache src/mmqa/__pycache__
