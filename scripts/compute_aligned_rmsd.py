"""Compute Kabsch-aligned RMSD for XYZ files.

Examples:
  Single pair:
    python compute_aligned_rmsd.py --output-xyz pred.xyz --reference-xyz ref.xyz --out-csv rmsd.csv

  Directory batch:
    python compute_aligned_rmsd.py --output-dir output_xyz --reference-dir reference_xyz --out-csv aligned_rmsd.csv

  Directory batch and write aligned XYZ:
    python compute_aligned_rmsd.py --output-dir output_xyz --reference-dir reference_xyz --out-csv aligned_rmsd.csv --write-aligned-dir aligned_output_xyz

This script depends only on the Python standard library and numpy. It does not
reorder atoms. Atom counts and element order must match.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable

import numpy as np


ROLE_SUFFIXES = (
    "_output",
    "_predicted",
    "_prediction",
    "_pred",
    "_reference",
    "_target",
    "_ref",
    "_input",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute Kabsch-aligned RMSD for XYZ files.")
    single = parser.add_argument_group("single-pair mode")
    single.add_argument("--output-xyz", type=Path, default=None, help="Predicted/output XYZ file.")
    single.add_argument("--reference-xyz", type=Path, default=None, help="Reference/target XYZ file.")
    batch = parser.add_argument_group("directory-batch mode")
    batch.add_argument("--output-dir", type=Path, default=None, help="Directory containing predicted/output XYZ files.")
    batch.add_argument("--reference-dir", type=Path, default=None, help="Directory containing reference/target XYZ files.")
    parser.add_argument("--out-csv", type=Path, required=True, help="Output CSV path.")
    parser.add_argument("--write-aligned-dir", type=Path, default=None, help="Optional directory for aligned output XYZ files.")
    return parser.parse_args()


def read_xyz(path: Path) -> tuple[list[str], np.ndarray, str]:
    lines = [line.rstrip("\n") for line in path.read_text(encoding="utf-8").splitlines()]
    nonempty = [line.strip() for line in lines if line.strip()]
    if len(nonempty) < 2:
        raise ValueError(f"Invalid XYZ file: {path}")
    try:
        n_atoms = int(nonempty[0])
    except ValueError as exc:
        raise ValueError(f"Invalid atom count in {path}: {nonempty[0]!r}") from exc
    comment = nonempty[1] if len(nonempty) > 1 else ""
    atom_lines = nonempty[2 : 2 + n_atoms]
    if len(atom_lines) != n_atoms:
        raise ValueError(f"Atom count mismatch in {path}: expected {n_atoms}, found {len(atom_lines)}")
    symbols: list[str] = []
    coords: list[list[float]] = []
    for line in atom_lines:
        fields = line.split()
        if len(fields) < 4:
            raise ValueError(f"Invalid XYZ atom line in {path}: {line!r}")
        symbols.append(fields[0])
        coords.append([float(fields[1]), float(fields[2]), float(fields[3])])
    return symbols, np.asarray(coords, dtype=np.float64), comment


def write_xyz(path: Path, symbols: list[str], coords: np.ndarray, comment: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{len(symbols)}\n")
        handle.write(f"{comment}\n")
        for symbol, xyz in zip(symbols, coords):
            handle.write(f"{symbol:<2} {xyz[0]: .10f} {xyz[1]: .10f} {xyz[2]: .10f}\n")


def kabsch_align(output_coords: np.ndarray, reference_coords: np.ndarray) -> np.ndarray:
    output_mean = output_coords.mean(axis=0, keepdims=True)
    reference_mean = reference_coords.mean(axis=0, keepdims=True)
    output_centered = output_coords - output_mean
    reference_centered = reference_coords - reference_mean
    covariance = output_centered.T @ reference_centered
    u_matrix, _singular_values, vt_matrix = np.linalg.svd(covariance, full_matrices=False)
    rotation = u_matrix @ vt_matrix
    if np.linalg.det(rotation) < 0:
        u_matrix = u_matrix.copy()
        u_matrix[:, -1] *= -1.0
        rotation = u_matrix @ vt_matrix
    translation = reference_mean - output_mean @ rotation
    return output_coords @ rotation + translation


def aligned_rmsd(output_coords: np.ndarray, reference_coords: np.ndarray) -> tuple[np.ndarray, float]:
    aligned = kabsch_align(output_coords, reference_coords)
    diff = aligned - reference_coords
    rmsd = math.sqrt(float(np.mean(np.sum(diff * diff, axis=1))))
    return aligned, rmsd


def normalized_stem(path: Path) -> str:
    stem = path.stem
    changed = True
    while changed:
        changed = False
        for suffix in ROLE_SUFFIXES:
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                changed = True
    return stem


def build_reference_index(reference_dir: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in sorted(reference_dir.glob("*.xyz")):
        for key in (path.name.lower(), path.stem.lower(), normalized_stem(path).lower()):
            index.setdefault(key, path)
    return index


def pair_directory(output_dir: Path, reference_dir: Path) -> list[tuple[str, Path, Path | None]]:
    reference_index = build_reference_index(reference_dir)
    pairs: list[tuple[str, Path, Path | None]] = []
    for output_path in sorted(output_dir.glob("*.xyz")):
        keys = (output_path.name.lower(), output_path.stem.lower(), normalized_stem(output_path).lower())
        reference_path = None
        for key in keys:
            if key in reference_index:
                reference_path = reference_index[key]
                break
        pairs.append((normalized_stem(output_path), output_path, reference_path))
    return pairs


def evaluate_pair(molecule_id: str, output_path: Path, reference_path: Path | None, aligned_dir: Path | None) -> dict[str, object]:
    row: dict[str, object] = {
        "molecule_id": molecule_id,
        "output_xyz": str(output_path),
        "reference_xyz": "" if reference_path is None else str(reference_path),
        "n_atoms": "",
        "aligned_rmsd_A": "",
        "status": "failed",
        "error_message": "",
    }
    try:
        if reference_path is None:
            raise FileNotFoundError(f"No matching reference XYZ for {output_path.name}")
        output_symbols, output_coords, _output_comment = read_xyz(output_path)
        reference_symbols, reference_coords, _reference_comment = read_xyz(reference_path)
        row["n_atoms"] = len(reference_symbols)
        if len(output_symbols) != len(reference_symbols):
            raise ValueError(f"Atom count mismatch: output={len(output_symbols)}, reference={len(reference_symbols)}")
        if output_symbols != reference_symbols:
            raise ValueError(f"Element order mismatch: output={output_symbols}, reference={reference_symbols}")
        aligned, rmsd = aligned_rmsd(output_coords, reference_coords)
        row["aligned_rmsd_A"] = f"{rmsd:.10f}"
        row["status"] = "success"
        if aligned_dir is not None:
            aligned_path = aligned_dir / output_path.name
            write_xyz(aligned_path, output_symbols, aligned, f"{molecule_id} Kabsch-aligned output")
    except Exception as exc:
        row["error_message"] = repr(exc)
    return row


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["molecule_id", "output_xyz", "reference_xyz", "n_atoms", "aligned_rmsd_A", "status", "error_message"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    single_mode = args.output_xyz is not None or args.reference_xyz is not None
    batch_mode = args.output_dir is not None or args.reference_dir is not None
    if single_mode == batch_mode:
        raise SystemExit("Choose exactly one mode: --output-xyz/--reference-xyz or --output-dir/--reference-dir")
    if single_mode:
        if args.output_xyz is None or args.reference_xyz is None:
            raise SystemExit("Single-pair mode requires both --output-xyz and --reference-xyz")
        rows = [evaluate_pair(normalized_stem(args.output_xyz), args.output_xyz, args.reference_xyz, args.write_aligned_dir)]
    else:
        if args.output_dir is None or args.reference_dir is None:
            raise SystemExit("Directory-batch mode requires both --output-dir and --reference-dir")
        pairs = pair_directory(args.output_dir, args.reference_dir)
        rows = [evaluate_pair(molecule_id, output_path, reference_path, args.write_aligned_dir) for molecule_id, output_path, reference_path in pairs]
    write_csv(args.out_csv, rows)


if __name__ == "__main__":
    main()
