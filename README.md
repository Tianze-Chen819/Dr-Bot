# Dr-Bot
— Patient Q&A LLM (Local)

## 1.Repo Structure
```
Dr-Bot/
├─ package/
│  ├─ __init__.py          # Model.predict(str) -> str
│  └─ assets/
│     └─ lora/             # created after training
├─ inference_cli.py        # terminal Q&A
├─ train_local_cuda.py     # QLoRA training (CUDA; MPS/CPU works but slower)
├─ requirements-cuda.txt   # NVIDIA setup
├─ training_data.csv       # Question,Physician Response
├─ baseline.ipynb
└─ Dr.Bot.pdf
```
## 2.Quick Start (Inference)

**macOS (MPS)***
```bash
cd Dr-Bot
python3 -m venv .venv && source .venv/bin/activate
pip install "torch>=2.3" transformers peft
export DRBOT_BASE="Qwen/Qwen2.5-3B-Instruct"   # or Qwen/Qwen2.5-1.5B-Instruct
python -u inference_cli.py
```
When you see Question>, type your question. Type exit to quit.

**NVIDIA (CUDA)**
```bash
cd Dr-Bot
conda create -n drbot python=3.11 -y
conda activate drbot
pip install -r requirements-cuda.txt
setx DRBOT_BASE "Qwen/Qwen2.5-3B-Instruct"     # on Windows
# macOS/Linux: export DRBOT_BASE="Qwen/Qwen2.5-3B-Instruct"
python -u inference_cli.py
```
