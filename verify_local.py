import torch
import sys
import json
import os
import random
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer
from peft import PeftModel
from selfies import decoder
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw

# --- 1. WINDOWS PATHS ---
project_root = Path(r"C:\Users\samyb\PythonProjects\MARL\drug-dreaming")
local_model_path = project_root / "models" / "Dream-v0-Base-7B"
tok_path = project_root / "models" / "dream_chemical_tokenizer"
# Updated to match the underscore from your error log
checkpoint_path = project_root / "outputs" / "dream_lora" / "checkpoint_600"

def generate_local(target_property=3.5):
    # --- 2. PATH SURGERY ---
    model_dir = os.path.abspath(local_model_path)
    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)

    for module in ["modeling_dream", "configuration_dream"]:
        if module in sys.modules:
            del sys.modules[module]

    import modeling_dream
    import configuration_dream

    # --- 3. ROPE SURGERY ---
    config_file = local_model_path / "config.json"
    if config_file.exists():
        with open(config_file, 'r') as f: config_data = json.load(f)
        if "rope_scaling" in config_data and isinstance(config_data["rope_scaling"], dict):
            config_data["rope_scaling"].setdefault("factor", 1.0)
        with open(config_file, 'w') as f: json.dump(config_data, f, indent=2)

    config_dir = os.path.abspath(local_model_path)
    config = configuration_dream.DreamConfig.from_pretrained(config_dir)
    if hasattr(config, 'rope_scaling') and isinstance(config.rope_scaling, dict):
        config.rope_scaling.setdefault('factor', 1.0)
    else:
        config.rope_scaling = {'type': 'linear', 'factor': 1.0}

    # --- 4. LOAD MODEL TO RAM ---
    print("--- Phase 1: Loading Base Model to RAM (Takes 1-2 mins) ---")
    base_model = modeling_dream.DreamModel.from_pretrained(
        config_dir,
        config=config,
        torch_dtype=torch.float32,
        device_map={"":"cpu"}
    )
    
    tok_dir = os.path.abspath(tok_path)
    tokenizer = AutoTokenizer.from_pretrained(tok_dir, trust_remote_code=True)
    base_model.resize_token_embeddings(len(tokenizer))
    
    # --- 5. ADAPTER WEIGHT AUTO-FIX & LOAD ---
    print("--- Phase 2: Attaching LoRA Adapters ---")
    
    adapter_config = checkpoint_path / "adapter_config.json"
    if not adapter_config.exists():
        raise FileNotFoundError(f"\nSTOP: Missing config at {adapter_config}")
    
    # Check for valid weights, fix if misnamed
    weight_st = checkpoint_path / "adapter_model.safetensors"
    weight_bin = checkpoint_path / "adapter_model.bin"
    
    if not weight_st.exists() and not weight_bin.exists():
        print("Valid weights file not found. Hunting for misnamed files...")
        candidates = list(checkpoint_path.glob("adapter_model*"))
        if candidates:
            misnamed_file = candidates[0]
            print(f"Found broken filename: {misnamed_file.name}. Auto-renaming to adapter_model.safetensors...")
            misnamed_file.rename(weight_st)
        else:
            raise FileNotFoundError(f"\nSTOP: The actual weights file is completely missing from {checkpoint_path}.\nPlease go back to Google Drive and ensure you download the massive weights file, not just the tiny JSON files.")

    # Load natively
    ckpt_dir = os.path.abspath(checkpoint_path)
    model = PeftModel.from_pretrained(base_model, ckpt_dir, local_files_only=True)
    model.eval()

    # --- 6. GENERATION LOOP ---
    print(f"--- Phase 3: Dreaming (Target LogP: {target_property}) ---")
    print("Generating on CPU. This will be slow, please wait...")
    
    input_ids = torch.full((1, 32), tokenizer.mask_token_id, device="cpu")
    cond = torch.tensor([[target_property]], dtype=torch.float32, device="cpu")
    
    with torch.no_grad():
        for t_val in torch.linspace(1, 0, 50): 
            step_t = t_val.repeat(1).unsqueeze(-1).to("cpu")
            outputs = model(input_ids=input_ids, timesteps=step_t, cond_values=cond)
            input_ids = torch.argmax(outputs.logits, dim=-1)
            print(".", end="", flush=True)
    print("\nGeneration Complete!")

    # --- 7. DECODE & VALIDATE ---
    selfies_str = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    print(f"\nResulting SELFIES: {selfies_str}")
    
    try:
        smiles = decoder(selfies_str)
        mol = Chem.MolFromSmiles(smiles)
        
        if mol:
            actual = Descriptors.MolLogP(mol)
            print(f"✅ Success! Chemically Valid.")
            print(f"SMILES: {smiles}")
            print(f"Calculated LogP: {actual:.2f}")
            img_path = "dreamed_molecule.png"
            Draw.MolToFile(mol, img_path)
            print(f"Structure saved to {img_path}")
        else:
            print(f"❌ Chemically Invalid SMILES: {smiles}")
    except Exception as e:
        print(f"❌ Error during chemical decoding: {e}")

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    generate_local(target_property=3.5)