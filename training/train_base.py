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

# --- STEP 1: CONDITIONAL DATASET (GROUND FLOOR BYPASS) ---
class ConditionalSELFIESDataset(Dataset):
    def __init__(self, selfies_path, property_path, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        # Grab the raw dictionary so we don't have to call any tokenizer methods
        self.vocab = tokenizer.get_vocab()
        self.unk_id = tokenizer.unk_token_id if tokenizer.unk_token_id is not None else 0
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        
        with open(selfies_path, 'r') as f:
            self.lines = [line.strip() for line in f if line.strip()]
        self.properties = torch.load(property_path)

    def __len__(self): return len(self.lines)

    def __getitem__(self, idx):
        # 1. Get raw string tokens
        tokens = self.tokenizer.tokenize(self.lines[idx])[:self.max_length]
        
        # 2. Manual dictionary lookup (The Absolute Bypass)
        # This is pure Python; it CANNOT trigger the library's padding bug.
        raw_ids = [self.vocab.get(t, self.unk_id) for t in tokens]
        input_ids = torch.tensor(raw_ids, dtype=torch.long)
        
        # 3. Manually pad with pure PyTorch
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

    def __call__(self, features):
        input_ids = torch.stack([f["input_ids"] for f in features])
        cond_values = torch.stack([f["cond_value"] for f in features]).float()
        labels = input_ids.clone()
        t = torch.rand(input_ids.shape[0])
        alpha_t = torch.cos((t * torch.pi / 2)) ** 2
        mask_map = torch.rand_like(input_ids.float()) > alpha_t.unsqueeze(-1)
        input_ids[mask_map] = self.mask_id
        labels[~mask_map] = -100
        return {
            "input_ids": input_ids,
            "labels": labels,
            "timesteps": t.unsqueeze(-1),
            "cond_values": cond_values.unsqueeze(-1)
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
                device = next(self.sample_model.parameters()).device
                input_ids = torch.full((len(self.targets), 32), self.tokenizer.mask_token_id, device=device)
                conds = self.targets.unsqueeze(-1).to(device)
                t_vals = torch.linspace(1, 0, 10).to(device)
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
    # --- 0. REPRODUCIBILITY ---
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    wandb.init(project="drug-dreaming-diffusion", entity="drug-discovery-dream")
    wandb.config.update({"seed": SEED})

    # Auto-detect environment
    if Path("/content").exists() and Path("/content/models").exists():
        project_root = Path("/content")                               # Colab local disk
    elif Path("/content").exists():
        project_root = Path("/content/drive/MyDrive/drug-dreaming")  # Colab Drive fallback
    else:
        project_root = Path(__file__).parent.parent                   # Local Windows

    local_model_path = project_root / "models" / "Dream-v0-Base-7B"
    tok_path = project_root / "models" / "dream_chemical_tokenizer"
    selfies_path = project_root / "data" / "zinc250k" / "zinc250k_selfies.txt"
    prop_path = project_root / "data" / "zinc250k" / "zinc250k_properties.pt"

    print(f"Using project_root: {project_root}")
    print(f"Model path: {local_model_path}")

    # --- 1. MEMORY PURGE ---
    for key in list(sys.modules.keys()):
        if "modeling_dream" in key or "configuration_dream" in key or "generation_utils" in key:
            del sys.modules[key]

    # --- 2. CONFIG SURGERY (NEUTRALIZE HUB) ---
    config_file = local_model_path / "config.json"
    if config_file.exists():
        with open(config_file, 'r') as f: config_data = json.load(f)
        config_data.pop("auto_map", None)
        if "rope_scaling" in config_data and isinstance(config_data["rope_scaling"], dict):
            config_data["rope_scaling"].setdefault("factor", 1.0)
        with open(config_file, 'w') as f: json.dump(config_data, f, indent=2)
        print(f"config.json rope_scaling: {config_data.get('rope_scaling', 'NOT FOUND')}")

    # --- 3. MODELING SURGERY (DE-DOTTING) ---
    local_modeling_file = local_model_path / "modeling_dream.py"
    if local_modeling_file.exists():
        print(f"Surgery: De-dotting and patching {local_modeling_file}...")
        with open(local_modeling_file, 'r') as f: content = f.read()
        content = re.sub(r'from \.([\w]+)', r'from \1', content)
        if "ROPE_INIT_FUNCTIONS.get" not in content:
            content = content.replace(
                "self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]",
                "self.rope_init_fn = ROPE_INIT_FUNCTIONS.get(self.rope_type, list(ROPE_INIT_FUNCTIONS.values())[0])"
            )
        with open(local_modeling_file, 'w') as f: f.write(content)

    # --- 3b. CONFIGURATION SURGERY (DE-DOTTING) ---
    local_config_file = local_model_path / "configuration_dream.py"
    if local_config_file.exists():
        print(f"Surgery: De-dotting {local_config_file}...")
        with open(local_config_file, 'r') as f: content = f.read()
        content = re.sub(r'from \.([\w]+)', r'from \1', content)
        with open(local_config_file, 'w') as f: f.write(content)

    # --- 4. CLEAN LOAD — QLoRA GPU MODE ---
    print("Forcing clean local load (QLoRA 4-bit)...")
    sys.path.insert(0, str(local_model_path))

    import modeling_dream
    import configuration_dream

    config = configuration_dream.DreamConfig.from_pretrained(local_model_path)

    print(f"Live config rope_scaling: {getattr(config, 'rope_scaling', 'ATTRIBUTE MISSING')}")
    if hasattr(config, 'rope_scaling') and isinstance(config.rope_scaling, dict):
        config.rope_scaling.setdefault('factor', 1.0)
        print(f"rope_scaling after fix: {config.rope_scaling}")
    elif not hasattr(config, 'rope_scaling') or config.rope_scaling is None:
        config.rope_scaling = {'type': 'linear', 'factor': 1.0}

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,  # T4 supports fp16, not bf16
    )

    # STEP A: Load directly to GPU 0 with 4-bit quant
    print("Loading model to GPU (4-bit)...")
    model = modeling_dream.DreamModel.from_pretrained(
        local_model_path,
        config=config,
        quantization_config=bnb_config,
        device_map={"": 0},
    )

    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)

    # Diagnose and fix missing pad token
    print(f"eos_token: {tokenizer.eos_token}, mask_token: {tokenizer.mask_token}, pad_token: {tokenizer.pad_token}")
    print(f"All special tokens: {tokenizer.all_special_tokens}")
    
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        elif tokenizer.mask_token is not None:
            tokenizer.pad_token = tokenizer.mask_token
            tokenizer.pad_token_id = tokenizer.mask_token_id
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    print(f"Pad token after fix: {tokenizer.pad_token}, id: {tokenizer.pad_token_id}")

    # STEP B: Resize + tie BEFORE LoRA
    model.resize_token_embeddings(len(tokenizer))
    if hasattr(model, 'tie_weights'):
        model.tie_weights()
    print(f"Embedding size after resize: {model.get_input_embeddings().weight.shape}")

    # STEP C: prepare + LoRA
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=None,  # DreamModel is diffusion, not CausalLM
    ))
    model.print_trainable_parameters()

    # --- 5. START TRAINING ---
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="/content/drive/MyDrive/drug-dreaming/outputs/dream_lora",
            per_device_train_batch_size=1,
            gradient_accumulation_steps=16,
            learning_rate=8e-5,
            weight_decay=0.01,
            fp16=True,
            bf16=False,
            logging_steps=10,
            save_steps=200,
            save_total_limit=2,
            report_to="wandb",
            remove_unused_columns=False,
            dataloader_num_workers=0,
            use_cpu=False,
            seed=SEED,
            data_seed=SEED,
            warmup_steps=100,
        ),
        train_dataset=ConditionalSELFIESDataset(selfies_path, prop_path, tokenizer),
        data_collator=ConditionalDiffusionCollator(tokenizer),
        callbacks=[ConditionalValidationCallback(tokenizer, model)]
    )

    print("Final Launch Sequence Initiated...")
    trainer.train()

if __name__ == "__main__":
    main()