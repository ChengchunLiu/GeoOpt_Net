# dataset.py
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from sklearn.model_selection import train_test_split

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms
from rdkit.Chem.Scaffolds import MurckoScaffold

import torch
from torch_geometric.data import InMemoryDataset, Data

from config import Config


class MyData(Data):
    def __inc__(self, key, value, store=None):
        if key in [
            "true_dist_indices", "true_angle_indices", "true_dihedral_indices",
            "init_dist_indices", "init_angle_indices", "init_dihedral_indices",
            "angle_edge_index", "dihedral_edge_index"
        ]:
            return self.num_nodes
        return super().__inc__(key, value, store)


class MolecularDataset(InMemoryDataset):
    def __init__(self, root, split="train", config: Config | None = None):
        self.split = split
        self.cfg = config or Config.from_dict({})

        self.ptable = Chem.GetPeriodicTable()
        self.atomic_numbers = {
            self.ptable.GetElementSymbol(i): i for i in range(1, 119)
        }

        super().__init__(root)

        from torch.serialization import safe_globals
        from torch_geometric.data.data import DataEdgeAttr

        try:
            with safe_globals([DataEdgeAttr]):
                self.data, self.slices = torch.load(
                    self.processed_paths[0],
                    weights_only=False
                )
        except Exception as e:
            print(f"[{split}] load processed failed: {e}, start process()...")
            self.process()
            with safe_globals([DataEdgeAttr]):
                self.data, self.slices = torch.load(
                    self.processed_paths[0],
                    weights_only=False
                )

    # ----------------- PyG 必需属性 -----------------
    @property
    def raw_file_names(self):
        return [os.path.basename(self.cfg.data.data_path)]

    @property
    def processed_file_names(self):
        return [f"processed_{self.split}.pt"]

    def download(self):
        pass

    # ----------------- 主处理流程 -----------------
    def process(self):
        full_df = pd.read_csv(self.cfg.data.data_path)
        data_list = []
        success_count, failure_count = 0, 0

        for _, row in tqdm(full_df.iterrows(),
                          total=len(full_df),
                          desc=f"Processing {self.split} from {self.cfg.data.theory_tag}"):
            try:
                data = self._process_molecule(row)
            except Exception as e:
                failure_count += 1
                print(f"Skip {row.get('smiles', 'N/A')} due to error: {e}")
                continue

            if data is None:
                failure_count += 1
                continue

            data_list.append(data)
            success_count += 1

        print(f"[{self.split}] success={success_count}, fail={failure_count}")
        if len(data_list) == 0:
            raise RuntimeError(f"No molecule processed for split={self.split}!")

        # 统计键长分布
        bond_stats = defaultdict(list)
        for data in data_list:
            atomic_nums = data.atomic_numbers.tolist()
            dist_indices = data.true_dist_indices.tolist()
            bond_types = data.true_bond_types
            if bond_types and isinstance(bond_types[0], list):
                bond_types = bond_types[0]
            bond_lengths = data.true_dists.tolist()

            for (i, j), btype, bl in zip(dist_indices, bond_types, bond_lengths):
                a = atomic_nums[i]
                b = atomic_nums[j]
                key = (min(a, b), max(a, b), float(btype))
                bond_stats[key].append(bl)

        quantile_thresholds = {}
        for key, lengths in bond_stats.items():
            lower = np.percentile(lengths, 0.1)
            upper = np.percentile(lengths, 99.9)
            quantile_thresholds[key] = (lower, upper)

        # 按键长分位数过滤异常分子
        filtered_data_list = []
        for data in data_list:
            atomic_nums = data.atomic_numbers.tolist()
            dist_indices = data.true_dist_indices.tolist()
            bond_types = data.true_bond_types
            if bond_types and isinstance(bond_types[0], list):
                bond_types = bond_types[0]
            bond_lengths = data.true_dists.tolist()

            remove = False
            for (i, j), btype, bl in zip(dist_indices, bond_types, bond_lengths):
                a = atomic_nums[i]
                b = atomic_nums[j]
                key = (min(a, b), max(a, b), float(btype))
                lower, upper = quantile_thresholds[key]
                if bl < lower or bl > upper:
                    remove = True
                    break
            if not remove:
                filtered_data_list.append(data)

        print(f"[{self.split}] filtered: {len(filtered_data_list)}/{len(data_list)}")

        # scaffold 分组
        scaffold_to_data = defaultdict(list)
        for data in filtered_data_list:
            bone = get_murcko_scaffold(data.smiles) or data.smiles
            scaffold_to_data[bone].append(data)

        train_list, val_list, test_list = split_by_scaffold(
            scaffold_to_data,
            threshold=10,
            seed=self.cfg.train.seed
        )

        os.makedirs(self.processed_dir, exist_ok=True)
        train_path = os.path.join(self.processed_dir, "processed_train.pt")
        val_path = os.path.join(self.processed_dir, "processed_val.pt")
        test_path = os.path.join(self.processed_dir, "processed_test.pt")

        torch.save(self.collate(train_list), train_path)
        torch.save(self.collate(val_list),   val_path)
        torch.save(self.collate(test_list),  test_path)

        scaffolds = sorted(scaffold_to_data.keys())
        scaffold2id = {s: idx for idx, s in enumerate(scaffolds)}

        def save_split(split_list, filename):
            records = []
            for data in split_list:
                sc = get_murcko_scaffold(data.smiles) or data.smiles
                records.append({
                    "smiles": data.smiles,
                    "scaffold": sc,
                    "scaffold_id": scaffold2id.get(sc, -1),
                    "domain_idx": int(data.domain_idx.item()),
                })
            df = pd.DataFrame(records)
            df.to_csv(os.path.join(self.root, filename), index=False)

        save_split(train_list, "train_split.csv")
        save_split(val_list,   "val_split.csv")
        save_split(test_list,  "test_split.csv")

        print(f"[{self.split}] saved train/val/test splits and PT files.")

    # ----------------- 单分子处理 -----------------
    def _process_molecule(self, row):
        cfg = self.cfg
        smiles = row["smiles"]
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("invalid SMILES")
        mol = Chem.AddHs(mol)

        try:
            Chem.rdPartialCharges.ComputeGasteigerCharges(mol)
        except Exception:
            pass

        params = AllChem.ETKDGv3()
        params.randomSeed = cfg.train.seed
        params.maxAttempts = 1000
        params.useSmallRingTorsions = True
        params.enforceChirality = True
        params.useRandomCoords = True

        try:
            ret = AllChem.EmbedMolecule(mol, params)
        except Exception as e:
            print(f"Embed error for {smiles}: {e}")
            return None
        if ret != 0 or mol.GetNumConformers() == 0:
            print(f"Embed failed for {smiles}, code={ret}")
            return None

        mmff_props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94")
        if mmff_props is None:
            raise RuntimeError(f"MMFF94 not supported: {smiles}")
        AllChem.MMFFOptimizeMolecule(mol, "MMFF94")

        conf = mol.GetConformer()

        atom_features = [self._get_atom_features(a) for a in mol.GetAtoms()]
        x = torch.tensor(atom_features, dtype=torch.float)

        atom_edge_index, edge_attr, edge_types, bond_types = self._get_bond_info(mol)
        edge_type = torch.tensor(edge_types, dtype=torch.long)
        pos = torch.tensor(
            [list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())],
            dtype=torch.float
        )

        dft_coords, _ = self._parse_coordinates(row["atom_coordinate"])
        if dft_coords.shape[0] != mol.GetNumAtoms():
            raise ValueError("DFT coords atom count mismatch")
        dft_pos = torch.tensor(dft_coords, dtype=torch.float)

        init_dists, init_angles, init_dihedrals, init_dist_idx, init_ang_idx, init_dih_idx = \
            self._compute_geometrics(mol, conf)
        true_dists, true_angles, true_dihedrals, true_dist_idx, true_ang_idx, true_dih_idx = \
            self._compute_geometrics_from_coords(mol, dft_coords)

        angle_edge_index, angle_edge_attr = self._construct_angle_edges(init_ang_idx, init_angles)
        dihedral_edge_index, dihedral_edge_attr = self._construct_dihedral_edges(init_dih_idx, init_dihedrals)

        data = MyData(
            atomic_numbers=torch.tensor(
                [a.GetAtomicNum() for a in mol.GetAtoms()],
                dtype=torch.long
            ),
            x=x,
            atom_edge_index=atom_edge_index,
            edge_attr=edge_attr,
            edge_type=edge_type,
            pos=pos,
            dft_pos=dft_pos,

            init_dists=init_dists,
            init_angles=init_angles,
            init_dihedrals=init_dihedrals,
            init_dist_indices=torch.tensor(init_dist_idx, dtype=torch.long),
            init_angle_indices=torch.tensor(init_ang_idx, dtype=torch.long),
            init_dihedral_indices=torch.tensor(init_dih_idx, dtype=torch.long),

            true_dists=true_dists,
            true_angles=true_angles,
            true_dihedrals=true_dihedrals,
            true_dist_indices=torch.tensor(true_dist_idx, dtype=torch.long),
            true_angle_indices=torch.tensor(true_ang_idx, dtype=torch.long),
            true_dihedral_indices=torch.tensor(true_dih_idx, dtype=torch.long),

            smiles=smiles,
            angle_edge_index=angle_edge_index,
            angle_edge_attr=angle_edge_attr,
            dihedral_edge_index=dihedral_edge_index,
            dihedral_edge_attr=dihedral_edge_attr,
            true_bond_types=bond_types,

            domain_idx=torch.tensor(cfg.data.domain_idx, dtype=torch.long)
        )
        return data

    # ----------------- 辅助函数 -----------------
    def _get_atom_features(self, atom):
        all_elems = list(self.atomic_numbers.keys())
        sym = atom.GetSymbol()
        elem_onehot = [1 if sym == e else 0 for e in all_elems]

        z = atom.GetAtomicNum()
        max_z = max(self.atomic_numbers.values())
        at_num_norm = z / max_z
        mass_norm = atom.GetMass() / 300.0

        degree = atom.GetDegree()
        total_valence = atom.GetTotalValence()

        hyb = atom.GetHybridization()
        hyb_types = [
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3
        ]
        hyb_onehot = [1 if hyb == t else 0 for t in hyb_types]
        hyb_onehot.append(0 if hyb in hyb_types else 1)

        in_ring = 1 if atom.IsInRing() else 0
        aromatic = 1 if atom.GetIsAromatic() else 0

        formal_charge = atom.GetFormalCharge()
        try:
            gcharge = float(atom.GetProp("_GasteigerCharge"))
        except Exception:
            gcharge = 0.0

        return (
            elem_onehot +
            [at_num_norm, mass_norm, degree, total_valence] +
            hyb_onehot +
            [in_ring, aromatic, formal_charge, gcharge]
        )

    def _get_bond_info(self, mol):
        bond_types_list = [
            Chem.rdchem.BondType.SINGLE,
            Chem.rdchem.BondType.DOUBLE,
            Chem.rdchem.BondType.TRIPLE,
            Chem.rdchem.BondType.AROMATIC
        ]
        stereo_list = [
            Chem.rdchem.BondStereo.STEREONONE,
            Chem.rdchem.BondStereo.STEREOZ,
            Chem.rdchem.BondStereo.STEREOE,
            Chem.rdchem.BondStereo.STEREOCIS,
            Chem.rdchem.BondStereo.STEREOTRANS
        ]

        edge_index, edge_attr, edge_type = [], [], []

        bonds = list(mol.GetBonds())
        M = len(bonds)

        for bond in bonds:
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            btype = bond.GetBondType()
            type_onehot = [1 if btype == bt else 0 for bt in bond_types_list]
            type_onehot.append(0 if btype in bond_types_list else 1)
            is_conj = 1.0 if bond.GetIsConjugated() else 0.0
            is_ring = 1.0 if bond.IsInRing() else 0.0
            is_arom = 1.0 if bond.GetIsAromatic() else 0.0
            stereo = bond.GetStereo()
            stereo_onehot = [1 if stereo == s else 0 for s in stereo_list]
            feat = type_onehot + [is_conj, is_ring, is_arom] + stereo_onehot

            edge_index.append([i, j])
            edge_index.append([j, i])
            edge_attr.append(feat)
            edge_attr.append(feat)

            if btype in bond_types_list:
                token = float(bond_types_list.index(btype))
            else:
                token = float(len(bond_types_list))
            edge_type.append(token)
            edge_type.append(token)

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        assert edge_index.size(1) == 2 * M

        true_bond_types = edge_type[0::2]
        assert len(true_bond_types) == M

        return edge_index, edge_attr, edge_type, true_bond_types

    def _parse_coordinates(self, coord_str):
        coord_str = coord_str.replace("\\n", "\n")
        atoms, coords = [], []
        for line in coord_str.strip().split("\n"):
            parts = line.split()
            if len(parts) == 4:
                atoms.append(parts[0])
                coords.append([float(x) for x in parts[1:]])
        return np.array(coords, dtype=np.float32), atoms

    def _compute_geometrics(self, mol, conf):
        dists, dist_indices = [], []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            pos_i = np.array(conf.GetAtomPosition(i))
            pos_j = np.array(conf.GetAtomPosition(j))
            dists.append(np.linalg.norm(pos_i - pos_j))
            dist_indices.append((min(i, j), max(i, j)))

        angles, angle_indices = [], []
        for atom in mol.GetAtoms():
            neigh = [n.GetIdx() for n in atom.GetNeighbors()]
            if len(neigh) < 2:
                continue
            for idx1 in range(len(neigh)):
                for idx2 in range(idx1 + 1, len(neigh)):
                    i, k = neigh[idx1], neigh[idx2]
                    j = atom.GetIdx()
                    pi = np.array(conf.GetAtomPosition(i))
                    pj = np.array(conf.GetAtomPosition(j))
                    pk = np.array(conf.GetAtomPosition(k))
                    v1 = pi - pj
                    v2 = pk - pj
                    cosang = np.dot(v1, v2) / (
                        np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8
                    )
                    angle = np.arccos(np.clip(cosang, -1.0, 1.0))
                    angles.append(angle)
                    angle_indices.append((i, j, k))

        dihedrals, dihedral_indices = [], []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            neigh_i = [n.GetIdx() for n in mol.GetAtomWithIdx(i).GetNeighbors() if n.GetIdx() != j]
            neigh_j = [n.GetIdx() for n in mol.GetAtomWithIdx(j).GetNeighbors() if n.GetIdx() != i]
            for k in neigh_i:
                for l in neigh_j:
                    angle_rad = np.deg2rad(rdMolTransforms.GetDihedralDeg(conf, k, i, j, l))
                    dihedrals.append(angle_rad)
                    dihedral_indices.append((k, i, j, l))

        return (
            torch.tensor(dists, dtype=torch.float),
            torch.tensor(angles, dtype=torch.float),
            torch.tensor(dihedrals, dtype=torch.float),
            dist_indices, angle_indices, dihedral_indices
        )

    def _compute_geometrics_from_coords(self, mol, coords):
        dists, dist_indices = [], []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            pi, pj = coords[i], coords[j]
            dists.append(np.linalg.norm(pi - pj))
            dist_indices.append((min(i, j), max(i, j)))

        angles, angle_indices = [], []
        for atom in mol.GetAtoms():
            neigh = [n.GetIdx() for n in atom.GetNeighbors()]
            if len(neigh) < 2:
                continue
            for idx1 in range(len(neigh)):
                for idx2 in range(idx1 + 1, len(neigh)):
                    i, k = neigh[idx1], neigh[idx2]
                    j = atom.GetIdx()
                    pi, pj, pk = coords[i], coords[j], coords[k]
                    v1 = pi - pj
                    v2 = pk - pj
                    cosang = np.dot(v1, v2) / (
                        np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8
                    )
                    angle = np.arccos(np.clip(cosang, -1.0, 1.0))
                    angles.append(angle)
                    angle_indices.append((i, j, k))

        dihedrals, dihedral_indices = [], []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            neigh_i = [n.GetIdx() for n in mol.GetAtomWithIdx(i).GetNeighbors() if n.GetIdx() != j]
            neigh_j = [n.GetIdx() for n in mol.GetAtomWithIdx(j).GetNeighbors() if n.GetIdx() != i]
            for k in neigh_i:
                for l in neigh_j:
                    angle_rad = self._calculate_dihedral(coords[k], coords[i], coords[j], coords[l])
                    dihedrals.append(angle_rad)
                    dihedral_indices.append((k, i, j, l))

        return (
            torch.tensor(dists, dtype=torch.float),
            torch.tensor(angles, dtype=torch.float),
            torch.tensor(dihedrals, dtype=torch.float),
            dist_indices, angle_indices, dihedral_indices
        )

    def _calculate_dihedral(self, p0, p1, p2, p3):
        b0 = p0 - p1
        b1 = p2 - p1
        b2 = p3 - p2
        b1_norm = b1 / (np.linalg.norm(b1) + 1e-6)
        v = b0 - np.dot(b0, b1_norm) * b1_norm
        w = b2 - np.dot(b2, b1_norm) * b1_norm
        return np.arctan2(
            np.dot(np.cross(b1_norm, v), w),
            np.dot(v, w)
        )

    def _construct_angle_edges(self, angle_indices, angles):
        angle_edge_index, angle_edge_attr = [], []
        for idx, (i, j, k) in enumerate(angle_indices):
            angle_edge_index.append([i, k])
            angle_edge_index.append([k, i])
            val = angles[idx].item()
            angle_edge_attr.append([val])
            angle_edge_attr.append([val])

        if angle_edge_index:
            return (
                torch.tensor(angle_edge_index, dtype=torch.long).t().contiguous(),
                torch.tensor(angle_edge_attr, dtype=torch.float)
            )
        else:
            return (
                torch.empty((2, 0), dtype=torch.long),
                torch.empty((0, 1), dtype=torch.float)
            )

    def _construct_dihedral_edges(self, dihedral_indices, dihedrals):
        dihedral_edge_index, dihedral_edge_attr = [], []
        for idx, (k, i, j, l) in enumerate(dihedral_indices):
            dihedral_edge_index.append([k, l])
            dihedral_edge_index.append([l, k])
            val = dihedrals[idx].item()
            dihedral_edge_attr.append([val])
            dihedral_edge_attr.append([val])

        if dihedral_edge_index:
            return (
                torch.tensor(dihedral_edge_index, dtype=torch.long).t().contiguous(),
                torch.tensor(dihedral_edge_attr, dtype=torch.float)
            )
        else:
            return (
                torch.empty((2, 0), dtype=torch.long),
                torch.empty((0, 1), dtype=torch.float)
            )


# ----------------- Scaffold 辅助函数 -----------------
def get_murcko_scaffold(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return MurckoScaffold.MurckoScaffoldSmiles(
        mol=mol,
        includeChirality=False
    )


def split_by_scaffold(scaffold_to_data: dict,
                      threshold: int = 10,
                      seed: int = 42):
    large_scaffolds = {s: lst for s, lst in scaffold_to_data.items() if len(lst) >= threshold}
    rare_data = []
    for s, lst in scaffold_to_data.items():
        if len(lst) < threshold:
            rare_data.extend(lst)

    train_list, val_list, test_list = [], [], []

    for scaffold, lst in large_scaffolds.items():
        tr, rest = train_test_split(lst, test_size=0.2, random_state=seed)
        vl, te = train_test_split(rest, test_size=0.5, random_state=seed)
        train_list.extend(tr)
        val_list.extend(vl)
        test_list.extend(te)

    if rare_data:
        tr, rest = train_test_split(rare_data, test_size=0.2, random_state=seed)
        vl, te = train_test_split(rest, test_size=0.5, random_state=seed)
        train_list.extend(tr)
        val_list.extend(vl)
        test_list.extend(te)

    return train_list, val_list, test_list
