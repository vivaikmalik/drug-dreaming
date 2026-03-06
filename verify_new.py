import selfies as sf
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
import numpy as np

def analyze_molecules(samples, targets, step):
    """
    Drop-in replacement for check_validity.
    Takes the 3 generated SELFIES, their conditioning targets, and the current training step.
    """
    
    table_rows = []
    
    for i, (selfies_str, target) in enumerate(zip(samples, targets)):
        print(f"\n--- Molecule {i+1} | Target={target} | Step={step} ---")
        print(f"SELFIES ({len(list(sf.split_selfies(selfies_str)))} tokens): {selfies_str}")
        
        row = {
            "step": step,
            "target": target,
            "selfies": selfies_str,
            "selfies_token_count": len(list(sf.split_selfies(selfies_str))),
        }
        
        try:
            smiles = sf.decoder(selfies_str)
            mol = Chem.MolFromSmiles(smiles)
            
            if mol is None:
                print("❌ INVALID after decoding")
                row.update({"smiles": smiles, "valid": False})
                table_rows.append(row)
                continue

            # --- Re-encode to measure how much was retained ---
            re_encoded = sf.encoder(smiles)
            reenc_token_count = len(list(sf.split_selfies(re_encoded)))
            retention = round(reenc_token_count / row["selfies_token_count"], 3)

            # --- Core properties ---
            logp    = Descriptors.MolLogP(mol)
            mw      = Descriptors.MolWt(mol)
            heavy   = mol.GetNumHeavyAtoms()
            rings   = rdMolDescriptors.CalcNumRings(mol)
            arom    = rdMolDescriptors.CalcNumAromaticRings(mol)
            hbd     = rdMolDescriptors.CalcNumHBD(mol)   # H-bond donors
            hba     = rdMolDescriptors.CalcNumHBA(mol)   # H-bond acceptors
            rot     = rdMolDescriptors.CalcNumRotatableBonds(mol)
            
            # --- Lipinski Rule of 5 (basic drug-likeness) ---
            lipinski = (mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10)

            # --- Canonical SMILES ---
            canonical_smiles = Chem.MolToSmiles(mol)

            print(f"SMILES:          {canonical_smiles}")
            print(f"Token retention: {retention:.0%}  ({reenc_token_count} re-encoded vs {row['selfies_token_count']} original)")
            print(f"MW={mw:.1f}  LogP={logp:.2f}  HBD={hbd}  HBA={hba}")
            print(f"HeavyAtoms={heavy}  Rings={rings}  AromaticRings={arom}  RotBonds={rot}")
            print(f"Lipinski Ro5: {'✅ PASS' if lipinski else '❌ FAIL'}")
            print(f"LogP vs Target delta: {abs(logp - target):.2f}")

            row.update({
                "smiles":           canonical_smiles,
                "valid":            True,
                "token_retention":  retention,
                "logp":             round(logp, 3),
                "mw":               round(mw, 2),
                "heavy_atoms":      heavy,
                "rings":            rings,
                "aromatic_rings":   arom,
                "hbd":              hbd,
                "hba":              hba,
                "rot_bonds":        rot,
                "lipinski_pass":    lipinski,
                "logp_target_delta": round(abs(logp - target), 3),
            })

        except Exception as e:
            print(f"❌ ERROR: {e}")
            row["valid"] = False

        table_rows.append(row)

    # --- Summary printout ---
    valid_rows = [r for r in table_rows if r.get("valid")]
    print(f"\n{'='*50}")
    print(f"SUMMARY  |  Step {step}  |  {len(valid_rows)}/3 valid")
    if valid_rows:
        mean_retention = np.mean([r["token_retention"] for r in valid_rows])
        mean_heavy     = np.mean([r["heavy_atoms"]     for r in valid_rows])
        mean_rings     = np.mean([r["rings"]           for r in valid_rows])
        mean_delta     = np.mean([r["logp_target_delta"] for r in valid_rows])
        print(f"Avg token retention : {mean_retention:.0%}   (want > 80%)")
        print(f"Avg heavy atoms     : {mean_heavy:.1f}       (want > 15)")
        print(f"Avg ring count      : {mean_rings:.1f}        (want > 1)")
        print(f"Avg LogP-target Δ   : {mean_delta:.2f}       (want < 1.0)")
    print(f"{'='*50}")

    return table_rows


# --- Example usage mirroring your current setup ---
step = 2500
targets = [1.0, 3.0, 5.0]
samples = [
    "[Ring2][Ring2][S@][=C][Ring1][C][C][C][C@@H1][C][P][C][Ring1][=Branch1][C][C][=C][C][Branch1][C][C][Ring1][C][=C][O][C][C][Branch1][C][O][Ring1][Ring1]",
    "[Ring2][NH3+1][N][C][#Branch1][Ring1][C][C][=Branch1][C][C][C][Ring1][Ring1][C@H1][C][C][Ring1][C][Ring1][=O][Ring1][=C][=N][Branch1][=N][C][C][C][Branch1][=C][Ring2]",
    "[O][Branch1][=Branch1][Ring1][C][C@H1][Branch1][C][C][Branch1][F][#Branch1][Branch1][Ring1][Branch1][Ring1][C][C][Ring1][#Branch1][C][=C][C][C][=O][C][C][Ring1][Branch1][=Branch1][C@@][N]"
]

rows = analyze_molecules(samples, targets, step)


# --- Example usage mirroring your current setup ---
step = 3000
targets = [1.0, 3.0, 5.0]
samples = [
    "[C][C][C][C][=Branch1][C][=C][C][=N][C][=Ring1][C][=C][C][Branch1][C][C][C][C][C][Ring1][C][C][C][C][C][N][NH1+1][C][C][C][C]",
    "[C][=C][Branch2][=C][C][C][C][C][C][C][#Branch1][=Branch1][C][C][C][C][C][C][C][C][=O][=N][C][C][C][Branch1][C][C][C][C][C][=Branch1]",
    "[N][=C][S][=O][#Branch1][C][C][=C][C][=C][C][O][#C][=N][C][C][=Branch1][C][C][Branch2][C][C][C][C][=C][C][C][C][C@@H1][C][C][C]"
]

rows = analyze_molecules(samples, targets, step)

# --- Example usage mirroring your current setup ---
step = 4000
targets = [1.0, 3.0, 5.0]
samples = [
    "[C][N][C][N][C][Ring1][O][C][C][O][=N][C][Ring1][=C][C][Ring1][=Branch1][=C][C][C][=Branch1][C][=Branch1][C@H1][C][Ring1][=Branch1][C][C][C][C][Branch1]",
    "[N][N][=C][C][=C][=C][=C][=C][C][N][=C][C][C][Ring1][C][Branch1][Branch1][Ring1][C][Ring1][Ring1][C][C][/C][=C][=N][C][C][=C][Branch1][Branch2][Ring1]",
    "[C][C][C][Ring1][C][C][=C][=C][N][Branch1][C][=C][Ring1][=C][C][C][Ring1][C][Branch1][=C][C][/N][N][=N][Ring1][C][C][C][=O][=C][=N][=C]"
]

rows = analyze_molecules(samples, targets, step)