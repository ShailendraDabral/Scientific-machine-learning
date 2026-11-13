"""Neural operators: DeepONet, Fourier Neural Operator and physics-informed neural operator."""
from .data import OperatorDataConfig, OperatorDataset, generate_dataset, sample_grf, solve_burgers
from .models import DeepONet, FNO2d, SpectralConv2d, count_parameters
from .physics import burgers_residual_grid
from .trainer import OperatorConfig, OperatorTrainer, build_model, evaluate_new_input, relative_l2

__all__ = ["DeepONet", "FNO2d", "SpectralConv2d", "OperatorConfig", "OperatorDataConfig", "OperatorDataset",
           "OperatorTrainer", "build_model", "burgers_residual_grid", "count_parameters", "evaluate_new_input", "generate_dataset",
           "relative_l2", "sample_grf", "solve_burgers"]
