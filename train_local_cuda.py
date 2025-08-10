"""
QLoRA finetune on NVIDIA (Windows/Linux). Produces adapters in package/assets/lora.
Edit CSV_PATH to point at your training_data.csv.
"""

import os, pandas as pd, torch
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
                          TrainingArguments, Trainer, DataCollatorForLanguageModeling)
from peft import LoraConfig, get_peft_model

# -------- Paths --------
CSV_PATH = r"./training_data.csv"          # <-- put your file here
OUTPUT_LORA_DIR = "./package/assets/lora"
BASE_ID = os.environ.get("DRBOT_BASE", "mistralai/Mistral-7B-Instruct-v0.2")
MAX_LEN = 1024

os.makedirs(OUTPUT_LORA_DIR, exist_ok=True)

# -------- Data --------
df = pd.read_csv(CSV_PATH,  encoding='latin1')
assert {"Question","Physician Response"}.issubset(df.columns)
df = df.dropna(subset=["Question","Physician Response"]).reset_index(drop=True)

SYSTEM = ("You are Dr. Bot, a patient-facing medical assistant. Be accurate, empathetic, "
          "safety-focused; use plain language; note uncertainty; avoid definitive diagnoses; "
          "suggest appropriate next steps and care settings.")

def format_row(q, a):
    return f"<s>[SYSTEM] {SYSTEM}\n[USER] {str(q).strip()}\n[ASSISTANT] {str(a).strip()}</s>"

df["text"] = [format_row(q, a) for q, a in zip(df["Question"], df["Physician Response"])]
ds = Dataset.from_pandas(df[["text"]])

# -------- Model/Tokenizer (4-bit) --------
bnb = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained(BASE_ID, use_fast=True, trust_remote_code=True)
if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_ID, device_map="auto", quantization_config=bnb,
    torch_dtype=torch.float16, trust_remote_code=True
)

# -------- LoRA --------
peft_cfg = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
)
model = get_peft_model(model, peft_cfg)

# -------- Tokenize --------
def tokenize(batch):
    out = tokenizer(batch["text"], max_length=MAX_LEN, truncation=True, padding=False)
    out["labels"] = out["input_ids"].copy()
    return out

tok_ds = ds.map(tokenize, batched=True, remove_columns=["text"])
collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# -------- Train --------
args = TrainingArguments(
    output_dir="./out",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    warmup_steps=50,
    logging_steps=25,
    save_strategy="epoch",
    evaluation_strategy="no",
    fp16=True,
    report_to="none",
)

trainer = Trainer(model=model, args=args, train_dataset=tok_ds, tokenizer=tokenizer, data_collator=collator)
trainer.train()

# -------- Save LoRA --------
model.save_pretrained(OUTPUT_LORA_DIR)
tokenizer.save_pretrained("./package/assets")
print(f"Saved adapters -> {OUTPUT_LORA_DIR}")
