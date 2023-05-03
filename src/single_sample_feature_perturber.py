import torch
import torch.nn as nn


class SingleSampleFeaturePerturber(nn.Module):
    def __init__(
        self,
        device: torch.device,
        feature_dims: tuple[int, int],
        perturbation_init_max: float = 0.001,
    ):
        super(SingleSampleFeaturePerturber, self).__init__()
        self._device = device
        self._feature_dims = feature_dims
        self.perturbation = nn.Parameter(
            torch.zeros(feature_dims, dtype=torch.float32)
        )
        self._perturbation_min = -1 * torch.ones(
            feature_dims, dtype=torch.float32
        )
        self._perturbation_max = torch.ones(feature_dims, dtype=torch.float32)
        self._perturbation_init_max = perturbation_init_max
        self.to(self._device)

    def reset_parameters(self, orig_feature: torch.tensor):
        assert orig_feature.shape == self.perturbation.shape
        if self.perturbation.grad is not None:
            self.perturbation.grad.zero_()
        self._perturbation_min = -1 * orig_feature
        self._perturbation_max = 1 - orig_feature
        unscaled_perturbation = 2 * torch.rand_like(orig_feature) - 1
        scaled_perturbation = (
            self._perturbation_init_max * unscaled_perturbation
        )
        self.perturbation.data = torch.clamp(
            scaled_perturbation,
            min=self._perturbation_min,
            max=self._perturbation_max,
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        return x + self.perturbation
