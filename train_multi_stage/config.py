# config.py
from dataclasses import dataclass, field, asdict
from typing import Dict
import torch


@dataclass
class DataConfig:
    data_path: str = "./data/raw_data/qm9_basis_opt_dataset_with_coords.csv"
    processed_dir: str = "./data/processed_data_scaffold_input_orient"
    save_dir: str = "./checkpoints"

    device: torch.device = field(
        default=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        repr=False
    )

    # 多理论标签：0/1/2...
    domain_idx: int = 0
    theory_tag: str = "B3LYP/6-31G(2df,p)"


@dataclass
class TrainConfig:
    batch_size: int = 32
    num_epochs: int = 200
    learning_rate: float = 1e-3
    seed: int = 42


@dataclass
class ModelConfig:
    hidden_dim: int = 256
    edge_feature_dim: int = 13
    max_num_nodes: int = 100

    num_atom_layers: int = 3
    num_angle_layers: int = 2
    num_dihedral_layers: int = 2
    gnn_dropout: float = 0.1

    transformer_num_layers: int = 2
    transformer_nhead: int = 8
    transformer_dropout: float = 0.1
    transformer_activation: str = "gelu"

    fc_hidden_dim: int = 128

    # 多理论设置
    n_domains: int = 3
    domain_embed_dim: int = 32
    infer_domain_idx: int = 2   # 推理默认使用的高理论 idx


@dataclass
class LossConfig:
    rmsd: float = 1.0
    bond: float = 0.5
    angle: float = 0.3
    dihedral: float = 0.2


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)

    def __post_init__(self):
        torch.manual_seed(self.train.seed)

        if self.model.max_num_nodes <= 0:
            raise ValueError("`model.max_num_nodes` must be > 0")
        if self.model.edge_feature_dim <= 0:
            raise ValueError("`model.edge_feature_dim` must be > 0")

        if self.model.transformer_activation not in ("relu", "gelu", "tanh"):
            print(f"Warning: unknown activation {self.model.transformer_activation}, use gelu.")
            self.model.transformer_activation = "gelu"

    @classmethod
    def from_dict(cls, cfg_dict: Dict) -> "Config":
        base = cls()

        data = {**asdict(base.data), **cfg_dict.get("data", {})}
        train = {**asdict(base.train), **cfg_dict.get("train", {})}
        model = {**asdict(base.model), **cfg_dict.get("model", {})}
        loss = {**asdict(base.loss), **cfg_dict.get("loss", {})}

        return cls(
            data=DataConfig(**data),
            train=TrainConfig(**train),
            model=ModelConfig(**model),
            loss=LossConfig(**loss),
        )


if __name__ == "__main__":
    cfg = Config.from_dict({})
    print(cfg)
