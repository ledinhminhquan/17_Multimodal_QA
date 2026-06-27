"""Generate the H100 AUTOPILOT Colab notebook as valid .ipynb JSON.

Run:  python notebooks/_build_notebook.py
Produces: notebooks/MMQA_Colab_Training_H100_AUTOPILOT.ipynb

Mirrors the resume-safe, GPU-auto-profiling, Colab-safe-install pattern proven in P02-P15.
Fine-tunes the ViLT VQA core on VQAv2.
"""

from __future__ import annotations

import json
from pathlib import Path

NB = "MMQA_Colab_Training_H100_AUTOPILOT.ipynb"


def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": list(lines)}


def code(*lines):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": list(lines)}


cells = []

cells.append(md(
    "# Multimodal Question Answering (VQA) - Colab Training (H100 AUTOPILOT, resume-safe)\n",
    "\n",
    "Fine-tunes the **ViLT VQA core** (`dandelin/vilt-b32-finetuned-vqa`) on VQAv2 with HF Trainer\n",
    "(soft-target BCE, selected on VQA accuracy). The question-type router, the calibrated-abstention\n",
    "gate and the answer constraints are algorithmic - only the VQA model is trained.\n",
    "\n",
    "**How to use:** set the controls in cell 0, then **Runtime -> Run all**. Resume-safe (re-run cell 9).\n",
    "Auto-adapts H100/A100/L4/T4. ViLT fine-tunes even on a free T4.\n",
    "\n",
    "> Low-confidence answers are returned as 'unsure' and flagged for human review.\n",
))

cells.append(code(
    "#@title 0) Controls - set these, then `Runtime -> Run all`  { display-mode: \"form\" }\n",
    "GIT_REPO_URL = \"https://github.com/<your-username>/mmqa\"  #@param {type:\"string\"}\n",
    "GIT_BRANCH   = \"main\"  #@param {type:\"string\"}\n",
    "USE_DRIVE    = True     #@param {type:\"boolean\"}\n",
    "DRIVE_SUBDIR = \"mmqa\"  #@param {type:\"string\"}\n",
    "\n",
    "VQA_BASE     = \"dandelin/vilt-b32-finetuned-vqa\"  #@param [\"dandelin/vilt-b32-finetuned-vqa\", \"dandelin/vilt-b32-mlm\"]\n",
    "VQA_DATASET  = \"HuggingFaceM4/VQAv2\"  #@param {type:\"string\"}\n",
    "EVAL_DATASET = \"lmms-lab/VQAv2\"  #@param {type:\"string\"}\n",
    "MAX_TRAIN_SAMPLES = 40000  #@param {type:\"integer\"}\n",
    "EPOCHS       = 4     #@param {type:\"integer\"}\n",
    "RUN_AUTOPILOT = True  #@param {type:\"boolean\"}\n",
    "HF_TOKEN     = \"\"      #@param {type:\"string\"}\n",
    "print('Controls set. VQA =', VQA_BASE, '| train =', VQA_DATASET)\n",
))

cells.append(code(
    "#@title 1) Check the GPU\n",
    "import subprocess\n",
    "print(subprocess.run(['nvidia-smi'], capture_output=True, text=True).stdout or 'No GPU - Runtime>Change runtime type>GPU')\n",
))

cells.append(code(
    "#@title 2) Mount Drive + artifact paths & HF caches  (BEFORE importing torch)\n",
    "import os\n",
    "ART = '/content/artifacts'\n",
    "if USE_DRIVE:\n",
    "    try:\n",
    "        from google.colab import drive\n",
    "        drive.mount('/content/drive')\n",
    "        ART = f'/content/drive/MyDrive/{DRIVE_SUBDIR}/artifacts'\n",
    "    except Exception as e:\n",
    "        print('Drive mount skipped:', e)\n",
    "os.makedirs(ART, exist_ok=True)\n",
    "os.environ['MMQA_ARTIFACTS_DIR'] = ART\n",
    "os.environ['HF_HOME'] = f'{ART}/hf_cache'\n",
    "os.makedirs(os.environ['HF_HOME'], exist_ok=True)\n",
    "if HF_TOKEN:\n",
    "    os.environ['HF_TOKEN'] = HF_TOKEN; os.environ['HUGGING_FACE_HUB_TOKEN'] = HF_TOKEN\n",
    "print('Artifacts ->', ART)\n",
))

cells.append(code(
    "#@title 3) Get the project source (git clone, or copy from Drive)\n",
    "import os\n",
    "os.chdir('/content')\n",
    "if os.path.isdir('/content/mmqa'):\n",
    "    os.chdir('/content/mmqa'); os.system('git pull')\n",
    "elif GIT_REPO_URL and '<your-username>' not in GIT_REPO_URL:\n",
    "    os.system(f'git clone -b {GIT_BRANCH} {GIT_REPO_URL} /content/mmqa'); os.chdir('/content/mmqa')\n",
    "else:\n",
    "    drive_src = f'/content/drive/MyDrive/{DRIVE_SUBDIR}/mmqa'\n",
    "    if os.path.isdir(drive_src):\n",
    "        os.system(f'cp -r {drive_src} /content/mmqa'); os.chdir('/content/mmqa')\n",
    "    else:\n",
    "        raise SystemExit('Set GIT_REPO_URL to your repo, or upload the project to Drive at ' + drive_src)\n",
    "print('cwd =', os.getcwd()); print(sorted(os.listdir('.'))[:20])\n",
))

cells.append(code(
    "#@title 4) Install dependencies (Colab-safe: NEVER reinstall torch)\n",
    "!pip -q install -r requirements_colab.txt\n",
    "!pip -q install -e . --no-deps\n",
    "print('deps installed')\n",
))

cells.append(code(
    "#@title 5) Verify environment + performance knobs (TF32)\n",
    "import torch\n",
    "print('torch', torch.__version__, '| CUDA', torch.cuda.is_available())\n",
    "if torch.cuda.is_available():\n",
    "    torch.backends.cuda.matmul.allow_tf32 = True\n",
    "    torch.backends.cudnn.allow_tf32 = True\n",
    "    print('GPU:', torch.cuda.get_device_name(0))\n",
    "import mmqa, transformers, datasets\n",
    "print('mmqa', mmqa.__version__, '| transformers', transformers.__version__)\n",
))

cells.append(code(
    "#@title 6) Auto GPU profile (ViLT VQA batch + precision)\n",
    "import torch\n",
    "name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'\n",
    "n = name.upper()\n",
    "if 'H100' in n:     BATCH = 64\n",
    "elif 'A100' in n:   BATCH = 32\n",
    "elif 'L4' in n:     BATCH = 16\n",
    "elif 'T4' in n:     BATCH = 12\n",
    "else:               BATCH = 4\n",
    "BF16 = torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False\n",
    "FP16 = (not BF16) and torch.cuda.is_available()\n",
    "TF32 = ('H100' in n or 'A100' in n or 'L4' in n)\n",
    "print(f'GPU={name} -> batch={BATCH} precision={\"bf16\" if BF16 else (\"fp16\" if FP16 else \"fp32\")}')\n",
))

cells.append(code(
    "#@title 7) Write the Colab training config  (configs/train_colab.yaml)\n",
    "import yaml, os\n",
    "cfg = {\n",
    "  'project_title': 'Multimodal Question Answering (VQA) System', 'author': 'Le Dinh Minh Quan', 'student_id': '23127460',\n",
    "  'data': {'vqa_dataset': VQA_DATASET, 'trust_remote_code': True, 'vqa_eval_dataset': EVAL_DATASET,\n",
    "           'vqa_eval_split': 'validation', 'use_hf': True, 'max_train_samples': int(MAX_TRAIN_SAMPLES),\n",
    "           'max_eval_samples': 4000, 'synth_eval_scenes': 120, 'image_size': 384, 'seed': 42},\n",
    "  'model': {'base_model': VQA_BASE, 'model_type': 'vilt', 'image_size': 384, 'max_question_length': 40,\n",
    "            'num_answers': 3129, 'top_k': 5, 'num_train_epochs': int(EPOCHS), 'learning_rate': 5.0e-5,\n",
    "            'per_device_train_batch_size': int(BATCH), 'warmup_ratio': 0.1, 'early_stopping_patience': 3,\n",
    "            'bf16': bool(BF16), 'fp16': bool(FP16), 'tf32': bool(TF32), 'eval_steps': 500, 'save_steps': 500},\n",
    "  'agent': {'confidence_min': 0.20, 'margin_min': 0.03, 'entropy_max': 2.5,\n",
    "            'abstain_enabled': True, 'type_consistency_enabled': True},\n",
    "}\n",
    "os.makedirs('configs', exist_ok=True)\n",
    "yaml.safe_dump(cfg, open('configs/train_colab.yaml','w'), sort_keys=False)\n",
    "print(open('configs/train_colab.yaml').read())\n",
))

cells.append(code(
    "#@title 8) Render synthetic scenes + sanity-check the datasets (streaming probe)\n",
    "!PYTHONPATH=src python -m mmqa.cli --config configs/train_colab.yaml gen-synthetic\n",
    "!PYTHONPATH=src python -m mmqa.cli --config configs/train_colab.yaml data\n",
))

cells.append(md(
    "## ONE BUTTON - autopilot (resume-safe)\n",
    "Persists the prior baseline, fine-tunes the ViLT VQA core, evaluates VQA accuracy + per-type +\n",
    "the agent's abstention/coverage, runs analysis, and writes **report.pdf + slides.pptx + grading +\n",
    "bundle**. Re-run to resume from the last checkpoint.\n",
))

cells.append(code(
    "#@title 9) ONE BUTTON autopilot  (re-run to resume)\n",
    "import os\n",
    "if RUN_AUTOPILOT:\n",
    "    os.system('PYTHONPATH=src python -m mmqa.cli --config configs/train_colab.yaml autopilot '\n",
    "              f'--limit {int(MAX_TRAIN_SAMPLES)}')\n",
    "else:\n",
    "    print('RUN_AUTOPILOT is off - use the individual steps below.')\n",
))

cells.append(md("## Individual steps (optional) - idempotent + resume-safe\n"))

cells.append(code(
    "#@title 10a) Fine-tune the ViLT VQA core (resumes from the last checkpoint)\n",
    "!PYTHONPATH=src python -m mmqa.cli --config configs/train_colab.yaml train-vqa --limit $MAX_TRAIN_SAMPLES --base-model \"$VQA_BASE\"\n",
))

cells.append(code(
    "#@title 10b) Baseline + evaluate (VQA accuracy + per-type + agent coverage) + tune\n",
    "!PYTHONPATH=src python -m mmqa.cli --config configs/train_colab.yaml train-baseline\n",
    "!PYTHONPATH=src python -m mmqa.cli --config configs/train_colab.yaml evaluate\n",
    "!PYTHONPATH=src python -m mmqa.cli --config configs/train_colab.yaml tune\n",
))

cells.append(code(
    "#@title 11) Diagnostics: eval headline + model metadata\n",
    "import json, glob, os\n",
    "rd = os.path.join(os.environ['MMQA_ARTIFACTS_DIR'], 'runs', 'eval.json')\n",
    "if os.path.exists(rd):\n",
    "    print(json.dumps(json.load(open(rd)).get('headline', {}), indent=2))\n",
    "for m in glob.glob(os.path.join(os.environ['MMQA_ARTIFACTS_DIR'], 'models', 'vqa', '*', 'model_meta.json')):\n",
    "    print(m); print(json.dumps(json.load(open(m)), indent=2)[:500])\n",
))

cells.append(md("## Test the trained model\n"))

cells.append(code(
    "#@title 12) Ask a question about the sample image with the trained model\n",
    "from PIL import Image\n",
    "import matplotlib.pyplot as plt\n",
    "!PYTHONPATH=src python -m mmqa.cli --config configs/train_colab.yaml ask --image sample_data/sample_scene.png --question \"how many shapes are there?\"\n",
    "plt.imshow(Image.open('sample_data/sample_scene.png')); plt.axis('off'); plt.title('sample image'); plt.show()\n",
))

cells.append(code(
    "#@title 13) Locate deliverables (report.pdf + slides.pptx + bundle)\n",
    "import glob, os\n",
    "base = os.environ['MMQA_ARTIFACTS_DIR']\n",
    "for pat in ['submission/*/report.pdf', 'submission/*/slides.pptx', 'submission/*/submission_bundle.zip']:\n",
    "    for f in glob.glob(os.path.join(base, pat)):\n",
    "        print(round(os.path.getsize(f)/1024, 1), 'KB', f)\n",
))

cells.append(code(
    "#@title 14) (Optional) Serve the API + Gradio demo\n",
    "# !PYTHONPATH=src python -m mmqa.cli --config configs/infer.yaml serve --ui --port 7860\n",
    "print('Uncomment to serve. On Colab add a tunnel (e.g. cloudflared) to expose :7860.')\n",
))

cells.append(md(
    "## Final checklist\n",
    "- [ ] GPU profile picked a sensible batch/precision\n",
    "- [ ] `train-vqa` wrote `models/vqa/<version>/`; `train-baseline` wrote `prior_baseline.json`\n",
    "- [ ] `evaluate` shows the fine-tuned ViLT beating the blind question-only prior on **VQA accuracy**\n",
    "- [ ] per-answer-type accuracy (yes/no, number, other) looks reasonable + abstention rate is sane\n",
    "- [ ] `ask` returns a sensible answer with a confidence; low-confidence -> 'unsure'\n",
    "- [ ] `report.pdf` + `slides.pptx` + `submission_bundle.zip` exist under `artifacts/submission/`\n",
    "- [ ] Remember: VQAv2 train mirror needs trust_remote_code; LLaVA/Qwen2.5-VL-3B are non-commercial (flagged)\n",
))


def main():
    nb = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "toc_visible": True},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }
    out = Path(__file__).resolve().parent / NB
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    json.loads(out.read_text(encoding="utf-8"))
    print(f"wrote {out}  ({len(cells)} cells)")


if __name__ == "__main__":
    main()
