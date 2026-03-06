import torch
import wandb
import sys
import re
import json
import random
import numpy as np
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

# --- STEP 2: CONDITIONAL DIFFUSION COLLATOR (PADDING-AWARE) ---
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
        
        # 1. Force padding tokens to NEVER be masked (stops the "cheating" mode)
        noise_probs.masked_fill_(pad_mask, 0.0)
        
        mask_map = noise_probs > alpha_t.unsqueeze(-1)
        input_ids[mask_map] = self.mask_id
        
        # 2. Set labels for unmasked or padding tokens to -100 so they are ignored by Loss
        labels[~mask_map] = -100
        labels[pad_mask] = -100
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "timesteps": t.unsqueeze(-1),
            "cond_values": cond_values.unsqueeze(-1)
        }

# --- STEP 3: CONDITIONAL VALIDATION CALLBACK (DTYPE FIXED) ---
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
                
                # Align dtypes with fp16 training to avoid Step 500 crash
                conds = self.targets.unsqueeze(-1).to(device, dtype=torch.float16)
                t_vals = torch.linspace(1, 0, 10).to(device, dtype=torch.float16)
                
                for t_val in t_vals:
                    step_t = t_val.repeat(len(self.targets)).unsqueeze(-1)
                    outputs = self.sample_model(input_ids=input_ids, timesteps=step_t, cond_values=conds)
                    input_ids = torch.argmax(outputs.logits, dim=-1)
                    
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

    if Path("/content").exists():
        project_root = Path("/content/drive/MyDrive/drug-dreaming")
    else:
        project_root = Path(__file__).parent.parent

    local_model_path = project_root / "models" / "Dream-v0-Base-7B"
    tok_path = project_root / "models" / "dream_chemical_tokenizer"
    selfies_path = project_root / "data" / "zinc250k" / "zinc250k_selfies.txt"
    prop_path = project_root / "data" / "zinc250k" / "zinc250k_properties.pt"

    # Load Config and Model with local surgeries applied
    sys.path.insert(0, str(local_model_path))
    import modeling_dream
    import configuration_dream

    config = configuration_dream.DreamConfig.from_pretrained(local_model_path)
    if not hasattr(config, 'rope_scaling') or config.rope_scaling is None:
        config.rope_scaling = {'type': 'linear', 'factor': 1.0}

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = modeling_dream.DreamModel.from_pretrained(
        local_model_path,
        config=config,
        quantization_config=bnb_config,
        device_map={"": 0},
    )

    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.resize_token_embeddings(len(tokenizer))
    if hasattr(model, 'tie_weights'): model.tie_weights()

    model = prepare_model_for_kbit_training(model)
    
    # 1. Higher Rank/Alpha for complex chemistry grammar
    model = get_peft_model(model, LoraConfig(
        r=64, 
        lora_alpha=128, 
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=None,
    ))
    model.print_trainable_parameters()

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(project_root / "outputs" / "dream_lora"),
            per_device_train_batch_size=1,
            gradient_accumulation_steps=16,
            learning_rate=2e-4, 
            weight_decay=0.01,
            fp16=True,
            logging_steps=10,
            save_steps=200,
            save_total_limit=2,
            report_to="wandb",
            remove_unused_columns=False,
            warmup_steps=100,
            
            # --- THE SAFETY ADDITIONS ---
            max_grad_norm=1.0,           # Surge protection for gradients
            gradient_checkpointing=True, # Saves VRAM, prevents OOM crashes
            lr_scheduler_type="cosine",  # Smoothly lowers LR at the end for better convergence
            # ----------------------------
        ),
        train_dataset=ConditionalSELFIESDataset(selfies_path, prop_path, tokenizer),
        data_collator=ConditionalDiffusionCollator(tokenizer),
        callbacks=[ConditionalValidationCallback(tokenizer, model)]
    )

    print("Breaking the 1.23 wall. Launching...")
    trainer.train()

if __name__ == "__main__":
    main()