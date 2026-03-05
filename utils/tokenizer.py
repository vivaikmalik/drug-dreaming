import os
import selfies as sf
from pathlib import Path
from transformers import AutoTokenizer

def build_chemical_vocabulary(selfies_path):
    """Parses the SELFIES dataset to extract the unique chemical alphabet."""
    print(f"Reading SELFIES from {selfies_path}...")
    
    with open(selfies_path, 'r') as f:
        selfies_list = [line.strip() for line in f if line.strip()]
    
    print("Extracting unique SELFIES alphabet...")
    alphabet = sorted(sf.get_alphabet_from_selfies(selfies_list))
    
    print(f"Found {len(alphabet)} unique chemical tokens.")
    return alphabet

def inject_tokens_and_save(base_model_path, custom_vocab, save_path):
    """Loads Dream 7B's base tokenizer, injects the alphabet, and saves locally."""
    print(f"Loading base tokenizer from {base_model_path}...")
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    
    print("Injecting chemical tokens into the vocabulary...")
    num_added = tokenizer.add_tokens(custom_vocab)
    print(f"Successfully added {num_added} new tokens.")
    
    print(f"Saving the customized chemical tokenizer to {save_path}...")
    tokenizer.save_pretrained(save_path)
    return tokenizer

def main():
    project_root = Path(__file__).parent.parent
    selfies_path = project_root / "data" / "zinc250k" / "zinc250k_selfies.txt"
    save_path = project_root / "models" / "dream_chemical_tokenizer"
    
    base_model_path = "Dream-org/Dream-v0-Base-7B" 
    
    save_path.mkdir(parents=True, exist_ok=True)
    
    if not selfies_path.exists():
        raise FileNotFoundError(f"SELFIES data not found at {selfies_path}. Run data/prep_zinc.py first.")
    
    custom_vocab = build_chemical_vocabulary(selfies_path)
    inject_tokens_and_save(base_model_path, custom_vocab, save_path)
    
    print("\nTokenization pipeline complete. The model is ready for Phase 1 training.")

if __name__ == "__main__":
    main()