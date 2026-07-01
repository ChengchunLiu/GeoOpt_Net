import argparse
import importlib
import json
import os
import sys

import numpy as np
import torch


def import_project_modules(project_root):
    project_root = os.path.abspath(project_root)
    candidate_dirs = [
        os.path.join(project_root, "train_multi_stage"),
        project_root,
    ]
    for directory in reversed(candidate_dirs):
        if os.path.isdir(directory) and directory not in sys.path:
            sys.path.insert(0, directory)

    config_module = importlib.import_module("config")
    model_module = importlib.import_module("model")
    dataset_module = importlib.import_module("dataset")

    Config = config_module.Config
    MultiGraphGeoGNNModel = model_module.MultiGraphGeoGNNModel
    build_single_data_from_smiles = dataset_module.build_single_data_from_smiles

    print("Loaded config from:", config_module.__file__)
    print("Loaded model from:", model_module.__file__)
    print("Loaded dataset from:", dataset_module.__file__)
    print("Using dataset builder:", "build_single_data_from_smiles")

    return Config, MultiGraphGeoGNNModel, build_single_data_from_smiles


def load_config(Config, config_json=None):
    if config_json is not None:
        with open(config_json, "r", encoding="utf-8") as handle:
            cfg_dict = json.load(handle)
        cfg = Config.from_dict(cfg_dict)
    else:
        cfg = Config.from_dict({})

    if hasattr(cfg, "model"):
        for key, value in vars(cfg.model).items():
            setattr(cfg, key, value)

    return cfg


def write_xyz(path, symbols, coords, title="refined geometry"):
    coords = np.asarray(coords, dtype=np.float64)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{len(symbols)}\n")
        handle.write(f"{title}\n")
        for symbol, xyz in zip(symbols, coords):
            handle.write(f"{symbol:2s} {xyz[0]: .10f} {xyz[1]: .10f} {xyz[2]: .10f}\n")


def resolve_device(device_arg):
    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "N/A"

    if device_arg == "cuda":
        if not cuda_available:
            raise RuntimeError("Requested --device cuda, but torch.cuda.is_available() is False")
        device = torch.device("cuda:0")
    elif device_arg == "auto":
        device = torch.device("cuda:0" if cuda_available else "cpu")
    else:
        device = torch.device(device_arg)

    print(f"torch.__version__: {torch.__version__}")
    print(f"torch.version.cuda: {torch.version.cuda}")
    print(f"torch.cuda.is_available(): {cuda_available}")
    print(f"torch.cuda.device_count(): {device_count}")
    print(f"torch.cuda.get_device_name(0): {device_name}")
    print(f"final device: {device}")

    return device


def load_checkpoint(model, checkpoint_path, device, allow_partial_load=False):
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state = checkpoint["state_dict"]
        else:
            state = checkpoint
    else:
        state = checkpoint

    model.load_state_dict(state, strict=not allow_partial_load)
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"strict checkpoint loading: {not allow_partial_load}")
    return model


def _data_symbols(data):
    symbols = getattr(data, "atom_symbols", None)
    if symbols is None:
        raise RuntimeError("Dataset builder did not attach atom_symbols to the inference Data object")
    return [str(symbol) for symbol in symbols]


def run_inference(args):
    project_root = os.path.abspath(args.project_root)
    Config, MultiGraphGeoGNNModel, build_single_data_from_smiles = import_project_modules(project_root)
    cfg = load_config(Config, args.config_json)
    device = resolve_device(args.device)

    data = build_single_data_from_smiles(
        smiles=args.smiles,
        input_xyz=args.input_xyz,
        atom_coordinate=None,
        domain_idx=args.domain_idx,
        random_seed=args.seed,
        require_target=False,
        config=cfg,
    )
    symbols = _data_symbols(data)
    initial_coords = data.pos.detach().cpu().numpy()

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
        write_xyz(args.output_initial_xyz, symbols, initial_coords, title="initial geometry")

    print(f"Saved refined geometry to: {args.output_xyz}")
    if args.output_initial_xyz is not None:
        print(f"Saved initial geometry to: {args.output_initial_xyz}")


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
    run_inference(parse_args())
