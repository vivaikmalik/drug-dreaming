import torch
import wandb
import sys
import re
import json
import random
import numpy as np
import os
import importlib.util
from pathlib import Path
from torch.utils.data import Dataset
from transformers import AutoTokenizer, Trainer, TrainingArguments, TrainerCallback, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# --- STEP 1: CONDITIONAL DATASET ---
class ConditionalSELFIESDataset(Dataset):
    def __init__(self, selfies_path, property_path, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.vocab = tokenizer.get_vocab()
        self.unk_id = tokenizer.unk_token_id if tokenizer.unk_token_id is not None else 0
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        
        with open(selfies_path, 'r') as f:
            self.lines = [line.strip() for line in f if line.strip()]
        self.properties = torch.load(property_path)

    def __len__(self): return len(self.lines)

    def __getitem__(self, idx):
        tokens = self.tokenizer.tokenize(self.lines[idx])[:self.max_length]
        raw_ids = [self.vocab.get(t, self.unk_id) for t in tokens]
        input_ids = torch.tensor(raw_ids, dtype=torch.long)
        
        seq_len = input_ids.shape[0]
        if seq_len < self.max_length:
            pad_tensor = torch.full((self.max_length - seq_len,), self.pad_id, dtype=input_ids.dtype)
            input_ids = torch.cat([input_ids, pad_tensor])
            
        return {
            "input_ids": input_ids,
            "cond_value": torch.tensor(float(self.properties[idx]))
        }

# --- STEP 2: CONDITIONAL DIFFUSION COLLATOR ---
class ConditionalDiffusionCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.mask_id = tokenizer.mask_token_id
        self.pad_id = tokenizer.pad_token_id

    def __call__(self, features):
        input_ids = torch.stack([f["input_ids"] for f in features])
        cond_values = torch.stack([f["cond_value"] for f in features]).float()
        labels = input_ids.clone()
        t = torch.rand(input_ids.shape[0])
        alpha_t = torch.cos((t * torch.pi / 2)) ** 2
        noise_probs = torch.rand_like(input_ids.float())
        pad_mask = (input_ids == self.pad_id)
        noise_probs.masked_fill_(pad_mask, 0.0)
        mask_map = noise_probs > alpha_t.unsqueeze(-1)
        input_ids[mask_map] = self.mask_id
        labels[~mask_map] = -100
        labels[pad_mask] = -100
        return {
            "input_ids": input_ids, "labels": labels, "timesteps": t.unsqueeze(-1), "cond_values": cond_values.unsqueeze(-1)
        }

# --- STEP 3: CONDITIONAL VALIDATION CALLBACK ---
class ConditionalValidationCallback(TrainerCallback):
    def __init__(self, tokenizer, model, property_targets=[1.0, 3.0, 5.0]):
        self.tokenizer = tokenizer
        self.sample_model = model
        self.targets = torch.tensor(property_targets)
        self.log_history = []

    def on_log(self, args, state, control, **kwargs):
        if state.global_step > 0 and state.global_step % 500 == 0:
            self.sample_model.eval()
            with torch.no_grad():
                device = "cuda" if torch.cuda.is_available() else "cpu"
                input_ids = torch.full((len(self.targets), 32), self.tokenizer.mask_token_id, device=device)
                conds = self.targets.unsqueeze(-1).to(device, dtype=torch.bfloat16)
                t_vals = torch.linspace(1, 0, 10).to(device, dtype=torch.bfloat16)
                for t_val in t_vals:
                    step_t = t_val.repeat(len(self.targets)).unsqueeze(-1)
                    outputs = self.sample_model(input_ids=input_ids, timesteps=step_t, cond_values=conds)
                    # Apply temperature to logits (e.g., 1.0 is standard, lower is safer, higher is crazier)
                    temperature = 1.0 
                    scaled_logits = (outputs.logits / temperature).to(torch.float32)
                    # Convert logits to probabilities
                    probs = torch.nn.functional.softmax(scaled_logits, dim=-1)
                    # Sample probabilistically handling the 3D tensor natively
                    input_ids = torch.distributions.Categorical(probs=probs).sample()

                for i in range(len(self.targets)):
                    decoded = self.tokenizer.decode(input_ids[i], skip_special_tokens=True)
                    self.log_history.append([state.global_step, self.targets[i].item(), decoded])
            wandb.log({"conditional_evolution": wandb.Table(columns=["Step", "Target", "SELFIES"], data=self.log_history)})
            self.sample_model.train()

# --- STEP 4: MAIN ---
def main():
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    wandb.init(project="drug-dreaming-diffusion", entity="drug-discovery-dream")

    local_model_path = Path("/content/models/Dream-v0-Base-7B")
    tok_path = Path("/content/models/dream_chemical_tokenizer")
    selfies_path = Path("/content/data/zinc250k/zinc250k_selfies.txt")
    prop_path = Path("/content/data/zinc250k/zinc250k_properties.pt")
    output_dir = "/content/drive/MyDrive/drug-dreaming/outputs/dream_lora"

    # --- A. FILE SURGERY (HARD OVERRIDE FOR ROPE) ---
    for filename in ["modeling_dream.py", "configuration_dream.py"]:
        f_path = local_model_path / filename
        if f_path.exists():
            with open(f_path, 'r') as f: content = f.read()
            
            # Fix 1: De-dotting relative imports
            content = re.sub(r'from \.([\w]+)', r'from \1', content)
            
            # Fix 2: Bypassing the failing ROPE init function entirely
            if filename == "modeling_dream.py":
                # Inject a safety check into the Rotary Embedding __init__
                if "self.rope_init_fn =" in content:
                    content = content.replace(
                        "inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, **self.rope_kwargs)",
                        "try:\n            inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, **self.rope_kwargs)\n        except Exception:\n            inv_freq, self.attention_scaling = ROPE_INIT_FUNCTIONS['linear'](self.config, device, factor=1.0)"
                    )
                # Ensure we don't crash on the 'default' key again
                content = content.replace(
                    "self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]",
                    "self.rope_init_fn = ROPE_INIT_FUNCTIONS.get(self.rope_type, ROPE_INIT_FUNCTIONS['linear'])"
                )

            with open(f_path, 'w') as f: f.write(content)

    # --- B. REGISTER MODULES GLOBALLY ---
    sys.path.insert(0, str(local_model_path))
    for mod_name in ["configuration_dream", "modeling_dream"]:
        spec = importlib.util.spec_from_file_location(mod_name, str(local_model_path / f"{mod_name}.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
    
    import modeling_dream
    import configuration_dream

    # --- C. INITIALIZE MODEL ---
    config = configuration_dream.DreamConfig.from_pretrained(local_model_path)
    
    # Critical: Use the exact structure required by modern Transformers
    config.rope_scaling = {"type": "linear", "factor": 1.0}
    
    model = modeling_dream.DreamModel.from_pretrained(
        local_model_path, config=config, torch_dtype=torch.bfloat16,attn_implementation="sdpa", device_map={"":0},
    )

    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model.resize_token_embeddings(len(tokenizer))
    if hasattr(model, 'tie_weights'): model.tie_weights()

    # --- D. LORA & TRAINER ---
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=64, lora_alpha=128, target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.1, bias="none", task_type=None,
    ))
    
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=output_dir, per_device_train_batch_size=8, gradient_accumulation_steps=8,
            learning_rate=4e-4, weight_decay=0.01, bf16=True, logging_steps=10, save_steps=200,
            save_total_limit=2, report_to="wandb", remove_unused_columns=False, warmup_steps=100,
            max_grad_norm=1.0, gradient_checkpointing=True, lr_scheduler_type="cosine",
        ),
        train_dataset=ConditionalSELFIESDataset(selfies_path, prop_path, tokenizer),
        data_collator=ConditionalDiffusionCollator(tokenizer),
        callbacks=[ConditionalValidationCallback(tokenizer, model)]
    )

    print("---LAUNCHING TRAINING ---")
    trainer.train()

if __name__ == "__main__":
    main()