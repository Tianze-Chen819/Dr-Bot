import os, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

_BASE_ID = os.environ.get("DRBOT_BASE", "Qwen/Qwen2.5-3B-Instruct")

class Model:
    def __init__(self):
        use_mps  = torch.backends.mps.is_available()
        use_cuda = torch.cuda.is_available()
        device   = "mps" if use_mps else ("cuda" if use_cuda else "cpu")
        dtype    = torch.float16 if device in {"cuda","mps"} else torch.float32

        print(f"[DrBot] Loading {_BASE_ID} on {device} …", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(_BASE_ID, use_fast=True, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(_BASE_ID, torch_dtype=dtype, trust_remote_code=True)
        base.to(device)

        lora_dir = os.path.join(os.path.dirname(__file__), "assets", "lora")
        if os.path.exists(os.path.join(lora_dir, "adapter_config.json")):
            self.model = PeftModel.from_pretrained(base, lora_dir)
            self.model.to(device)
        else:
            self.model = base

        self.device = device
        self.model.eval()
        print("[DrBot] Ready.", flush=True)

    @torch.inference_mode()
    def predict(self, question: str) -> str:
        print("[DrBot] Generating…", flush=True)
        prompt = f"<s>[SYSTEM] You are Dr. Bot…\n[USER] {question.strip()}\n[ASSISTANT]"
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        out = self.model.generate(
            **inputs,
            max_new_tokens=160,
            do_sample=True, temperature=0.7, top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )[0]

        text = self.tokenizer.decode(out, skip_special_tokens=True)
        if "[ASSISTANT]" in text:
            text = text.split("[ASSISTANT]", 1)[-1].strip()
        return text
