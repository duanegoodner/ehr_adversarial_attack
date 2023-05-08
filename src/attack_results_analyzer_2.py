import dill
import numpy as np
import torch as torch
from functools import cached_property
from pathlib import Path
from adv_attack import AdversarialExamplesSummary


def import_pickle_to_adv_example_summary(
    path: Path,
) -> AdversarialExamplesSummary:
    with path.open(mode="rb") as p:
        result = dill.load(p)
    return result





class AttackResultsAnalyzer:
    def __init__(self, result_path: Path):
        self.result_path = result_path
        self.result = import_pickle_to_adv_example_summary(result_path)

    @cached_property
    def orig_zeros_indices(self) -> np.ndarray:
        return np.where(self.result.orig_labels == 0)[0]

    @cached_property
    def orig_ones_indices(self) -> np.ndarray:
        return np.where(self.result.orig_labels == 1)[0]

    @cached_property
    def num_oz_perts(self) -> int:
        return len(self.orig_ones_indices)

    @cached_property
    def num_zo_perts(self) -> int:
        return len(self.orig_zeros_indices)

    @cached_property
    def oz_perts(self) -> torch.tensor:
        return self.result.perturbations[self.orig_ones_indices, :, :]

    @cached_property
    def zo_perts(self) -> torch.tensor:
        return self.result.perturbations[self.orig_zeros_indices, :, :]

    @cached_property
    def abs_oz_perts(self) -> torch.tensor:
        return torch.abs(self.oz_perts)

    @cached_property
    def abs_zo_perts(self) -> torch.tensor:
        return torch.abs(self.zo_perts)

    @cached_property
    def gmp_ij_oz(self) -> torch.tensor:
        return torch.max(self.abs_oz_perts, dim=0).values

    @cached_property
    def gmp_ij_zo(self) -> torch.tensor:
        return torch.max(self.abs_zo_perts, dim=0).values

    @cached_property
    def gap_ij_oz(self) -> torch.tensor:
        return torch.sum(self.abs_oz_perts, dim=0) / self.num_oz_perts

    @cached_property
    def gap_ij_zo(self) -> torch.tensor:
        return torch.sum(self.abs_zo_perts, dim=0) / self.num_zo_perts

    @cached_property
    def gpp_ij_oz(self) -> torch.tensor:
        return torch.norm(self.abs_oz_perts, p=1,  dim=0) / self.num_oz_perts

    @cached_property
    def gpp_ij_zo(self) -> torch.tensor:
        return torch.norm(self.abs_zo_perts, p=1,  dim=0) / self.num_zo_perts

    @cached_property
    def s_ij_oz(self) -> torch.tensor:
        return self.gmp_ij_oz * self.gpp_ij_oz

    @cached_property
    def s_ij_zo(self) -> torch.tensor:
        return self.gmp_ij_zo * self.gpp_ij_zo

    @cached_property
    def s_j_oz(self) -> torch.tensor:
        return torch.sum(self.s_ij_oz, dim=1)

    @cached_property
    def s_j_zo(self) -> torch.tensor:
        return torch.sum(self.s_ij_zo, dim=1)


analyzer = AttackResultsAnalyzer(
    result_path=Path(__file__).parent.parent
    / "data"
    / "k0.0-l10.15-lr0.1-ma100-ms1-2023-05-08_12:50:56.385216.pickle"
)
