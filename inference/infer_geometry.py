import os
import sys
import json
import argparse
import numpy as np
import torch
from torch_geometric.data import Data

from rdkit import Chem
from rdkit.Chem import AllChem

import importlib

def load_module_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def find_file(candidate_dirs, filename):
    for d in candidate_dirs:
        path = os.path.join(d, filename)
        if os.path.isfile(path):
            return path
    return None


def import_project_modules(project_root):
    project_root = os.path.abspath(project_root)

    candidate_dirs = [
        os.path.join(project_root, "train_multi_stage"),
        project_root,
    ]

    for d in reversed(candidate_dirs):
        if os.path.isdir(d) and d not in sys.path:
            sys.path.insert(0, d)

    config_module = importlib.import_module("config")
    model_module = importlib.import_module("model_20260522")

    Config = config_module.Config
    MultiGraphGeoGNNModel = model_module.MultiGraphGeoGNNModel

    try:
        dataset_module = importlib.import_module("dataset")
        MolecularDataset = getattr(dataset_module, "MolecularDataset", None)
    except ImportError:
        MolecularDataset = None

    print("Loaded config from:", config_module.__file__)
    print("Loaded model from:", model_module.__file__)

    if MolecularDataset is not None:
        print("Loaded dataset from:", dataset_module.__file__)

    return Config, MultiGraphGeoGNNModel, MolecularDataset


def load_config(Config, config_json=None):
    if config_json is not None:
        with open(config_json, "r", encoding="utf-8") as f:
            cfg_dict = json.load(f)
        cfg = Config.from_dict(cfg_dict)
    else:
        cfg = Config.from_dict({})

    if hasattr(cfg, "model"):
        for k, v in vars(cfg.model).items():
            setattr(cfg, k, v)

    return cfg


def read_xyz(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    n_atoms = int(lines[0])
    atom_lines = lines[2:2 + n_atoms]

    symbols = []
    coords = []

    for line in atom_lines:
        parts = line.split()
        symbols.append(parts[0])
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])

    return symbols, np.asarray(coords, dtype=np.float32)


def write_xyz(path, symbols, coords, title="refined geometry"):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{len(symbols)}\n")
        f.write(f"{title}\n")
        for s, xyz in zip(symbols, coords):
            f.write(f"{s:2s} {xyz[0]: .10f} {xyz[1]: .10f} {xyz[2]: .10f}\n")


def make_mol_from_smiles(smiles, xyz_symbols=None):
    mol0 = Chem.MolFromSmiles(smiles)
    if mol0 is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    candidates = [Chem.AddHs(mol0), mol0]

    if xyz_symbols is None:
        mol = candidates[0]
        Chem.SanitizeMol(mol)
        return mol

    for mol in candidates:
        Chem.SanitizeMol(mol)
        mol_symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
        if mol_symbols == xyz_symbols:
            return mol

    candidate_info = []
    for mol in candidates:
        candidate_info.append([atom.GetSymbol() for atom in mol.GetAtoms()])

    raise ValueError(
        "Atom order or atom count in XYZ does not match SMILES. "
        f"XYZ symbols: {xyz_symbols}; candidate SMILES symbols: {candidate_info}"
    )


def generate_initial_geometry(mol, seed=2026):
    mol = Chem.Mol(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)

    status = AllChem.EmbedMolecule(mol, params)
    if status != 0:
        status = AllChem.EmbedMolecule(mol, randomSeed=int(seed), useRandomCoords=True)

    if status != 0:
        raise RuntimeError("RDKit failed to generate an initial 3D geometry.")

    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    except Exception:
        try:
            AllChem.UFFOptimizeMolecule(mol, maxIters=500)
        except Exception:
            pass

    conf = mol.GetConformer()
    coords = []
    for i in range(mol.GetNumAtoms()):
        p = conf.GetAtomPosition(i)
        coords.append([p.x, p.y, p.z])

    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    return symbols, np.asarray(coords, dtype=np.float32)


def init_dataset_helper(MolecularDataset):
    if MolecularDataset is None:
        return None

    ds = MolecularDataset.__new__(MolecularDataset)
    ds.ptable = Chem.GetPeriodicTable()
    ds.atomic_numbers = {
        ds.ptable.GetElementSymbol(i): i
        for i in range(1, 119)
    }
    return ds


def fallback_atom_features(mol):
    feats = []
    for atom in mol.GetAtoms():
        feats.append([
            float(atom.GetAtomicNum()),
            float(atom.GetDegree()),
            float(atom.GetFormalCharge()),
            float(atom.GetTotalNumHs()),
            float(atom.GetIsAromatic()),
        ])
    return torch.tensor(feats, dtype=torch.float)


def bond_feature_vector(bond, dim):
    bt = bond.GetBondType()
    base = [
        float(bt == Chem.BondType.SINGLE),
        float(bt == Chem.BondType.DOUBLE),
        float(bt == Chem.BondType.TRIPLE),
        float(bt == Chem.BondType.AROMATIC),
        float(bond.GetIsConjugated()),
        float(bond.IsInRing()),
    ]

    if len(base) < dim:
        base = base + [0.0] * (dim - len(base))
    else:
        base = base[:dim]

    return base


def fallback_bond_info(mol, edge_feature_dim):
    edges = []
    attrs = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        feat = bond_feature_vector(bond, edge_feature_dim)

        edges.append([i, j])
        edges.append([j, i])
        attrs.append(feat)
        attrs.append(feat)

    if len(edges) == 0:
        return (
            torch.empty((2, 0), dtype=torch.long),
            torch.empty((0, edge_feature_dim), dtype=torch.float),
        )

    return (
        torch.tensor(edges, dtype=torch.long).t().contiguous(),
        torch.tensor(attrs, dtype=torch.float),
    )


def empty_edge_attr():
    return (
        torch.empty((2, 0), dtype=torch.long),
        torch.empty((0, 1), dtype=torch.float),
    )


def call_dataset_method(ds, names, mol):
    if ds is None:
        return None

    for name in names:
        if hasattr(ds, name):
            method = getattr(ds, name)
            try:
                result = method(mol)
                if isinstance(result, tuple) and len(result) >= 2:
                    return result[0], result[1]
            except TypeError:
                continue
            except Exception:
                continue

    return None


def to_tensor_pair(pair, attr_dim=1):
    if pair is None:
        return empty_edge_attr()

    edge_index, edge_attr = pair

    if not torch.is_tensor(edge_index):
        edge_index = torch.tensor(edge_index, dtype=torch.long)
    else:
        edge_index = edge_index.long()

    if edge_index.dim() == 2 and edge_index.size(0) != 2 and edge_index.size(1) == 2:
        edge_index = edge_index.t().contiguous()

    if not torch.is_tensor(edge_attr):
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    else:
        edge_attr = edge_attr.float()

    if edge_attr.dim() == 1:
        edge_attr = edge_attr.view(-1, 1)

    if edge_index.numel() == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, attr_dim), dtype=torch.float)

    return edge_index, edge_attr


def build_data(mol, coords, cfg, MolecularDataset, domain_idx):
    ds = init_dataset_helper(MolecularDataset)

    if ds is not None and hasattr(ds, "_get_atom_features"):
        try:
            x = torch.tensor(
                [ds._get_atom_features(atom) for atom in mol.GetAtoms()],
                dtype=torch.float,
            )
        except Exception:
            x = fallback_atom_features(mol)
    else:
        x = fallback_atom_features(mol)

    if ds is not None and hasattr(ds, "_get_bond_info"):
        try:
            bond_result = ds._get_bond_info(mol)
            atom_edge_index, edge_attr = bond_result[0], bond_result[1]
            atom_edge_index, edge_attr = to_tensor_pair(
                (atom_edge_index, edge_attr),
                attr_dim=getattr(cfg, "edge_feature_dim", edge_attr.size(-1) if torch.is_tensor(edge_attr) else 1),
            )
        except Exception:
            atom_edge_index, edge_attr = fallback_bond_info(
                mol,
                getattr(cfg, "edge_feature_dim", 6),
            )
    else:
        atom_edge_index, edge_attr = fallback_bond_info(
            mol,
            getattr(cfg, "edge_feature_dim", 6),
        )

    angle_pair = call_dataset_method(
        ds,
        [
            "_get_angle_info",
            "_get_angle_edge_info",
            "_get_angle_edges",
            "_get_angle_graph",
        ],
        mol,
    )
    angle_edge_index, angle_edge_attr = to_tensor_pair(angle_pair, attr_dim=1)

    dihedral_pair = call_dataset_method(
        ds,
        [
            "_get_dihedral_info",
            "_get_dihedral_edge_info",
            "_get_dihedral_edges",
            "_get_dihedral_graph",
        ],
        mol,
    )
    dihedral_edge_index, dihedral_edge_attr = to_tensor_pair(dihedral_pair, attr_dim=1)

    pos = torch.tensor(coords, dtype=torch.float)

    data = Data(
        x=x,
        pos=pos,
        atom_edge_index=atom_edge_index,
        edge_attr=edge_attr,
        angle_edge_index=angle_edge_index,
        angle_edge_attr=angle_edge_attr,
        dihedral_edge_index=dihedral_edge_index,
        dihedral_edge_attr=dihedral_edge_attr,
        batch=torch.zeros(pos.size(0), dtype=torch.long),
        domain_idx=torch.tensor(domain_idx, dtype=torch.long),
        num_nodes=pos.size(0),
    )

    return data


def load_checkpoint(model, checkpoint_path, device, allow_partial_load=False):
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location=device)

    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
        else:
            state = ckpt
    else:
        state = ckpt

    model.load_state_dict(state, strict=not allow_partial_load)
    return model


def run_inference(args):
    project_root = os.path.abspath(args.project_root)
    Config, MultiGraphGeoGNNModel, MolecularDataset = import_project_modules(project_root)

    cfg = load_config(Config, args.config_json)

    device = torch.device(args.device)

    if args.input_xyz is not None:
        xyz_symbols, coords = read_xyz(args.input_xyz)
        mol = make_mol_from_smiles(args.smiles, xyz_symbols)
        symbols = xyz_symbols
    else:
        mol = make_mol_from_smiles(args.smiles)
        symbols, coords = generate_initial_geometry(mol, seed=args.seed)

    data = build_data(
        mol=mol,
        coords=coords,
        cfg=cfg,
        MolecularDataset=MolecularDataset,
        domain_idx=args.domain_idx,
    )

    model = MultiGraphGeoGNNModel(cfg)
    model = load_checkpoint(
        model,
        args.checkpoint,
        device,
        allow_partial_load=args.allow_partial_load,
    )
    model.to(device)
    model.eval()

    data = data.to(device)

    with torch.no_grad():
        pred = model(data)

    pred_coords = pred.detach().cpu().numpy()

    os.makedirs(os.path.dirname(os.path.abspath(args.output_xyz)), exist_ok=True)
    write_xyz(args.output_xyz, symbols, pred_coords, title="refined geometry")

    if args.output_initial_xyz is not None:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_initial_xyz)), exist_ok=True)
        write_xyz(args.output_initial_xyz, symbols, coords, title="initial geometry")

    print(f"Saved refined geometry to: {args.output_xyz}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--smiles", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_xyz", type=str, required=True)

    parser.add_argument("--input_xyz", type=str, default=None)
    parser.add_argument("--output_initial_xyz", type=str, default=None)

    parser.add_argument("--project_root", type=str, default=".")
    parser.add_argument("--config_json", type=str, default=None)

    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--domain_idx", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)

    parser.add_argument("--allow_partial_load", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)