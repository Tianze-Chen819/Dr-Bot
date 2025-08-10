# Dr-Bot
— Patient Q&A LLM (Local)

Repo Structure
bash
perl```
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
