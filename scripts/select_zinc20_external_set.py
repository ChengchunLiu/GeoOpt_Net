import os
import sys
from rdkit import Chem

input_file_path = sys.argv[1]
output_file_path = sys.argv[2]

def filter_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    allowed = {"C","N","O","F", 'S', 'P', 'Si', 'F', 'Cl', 'Br', 'I', 'B'}
    for atom in mol.GetAtoms():
        if atom.GetSymbol() not in allowed:
            return None

    for atom in mol.GetAtoms():
        atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)

    Chem.AssignStereochemistry(mol, force=True, cleanIt=True)

    kekule = Chem.MolToSmiles(
        mol,
        isomericSmiles=True,  
        kekuleSmiles=True      
    )
    return kekule

unique_smiles = set()
with open(input_file_path, 'r') as infile:
    for line in infile:
        parts = line.strip().split()
        if not parts:
            continue
        raw = parts[0]
        out = filter_smiles(raw)
        if out:
            unique_smiles.add(out)

with open(output_file_path, 'w') as outfile:
    for smi in sorted(unique_smiles):
        outfile.write(smi + "\n")
