import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch
import torch.nn as nn
import seaborn as sns
from functools import cached_property
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from cv_trainer import WeightedRandomSamplerBuilder
from x19_mort_dataset import X19MortalityDataset


class X19MortalitySamplePlotter:
    def __init__(
        self,
        x19_mortality_dataset: Dataset = X19MortalityDataset(),
        # batch_size: int = 4,
        weighted_random_sampling: bool = True,
        model: nn.Module = None,
    ):
        self._dataset = x19_mortality_dataset
        # self._batch_size = batch_size
        self._weighted_random_sampling = weighted_random_sampling
        self._model = model
        self._sample_iterator = iter(self._dataloader)

    @cached_property
    def _sampler(self) -> WeightedRandomSampler | None:
        if self._weighted_random_sampling:
            return WeightedRandomSamplerBuilder(
                skewed_features=self._dataset[:][1]
            ).build()
        else:
            return None

    @cached_property
    def _dataloader(self) -> DataLoader:
        if self._sampler:
            plot_dataloader = DataLoader(
                dataset=self._dataset, batch_size=1, sampler=self._sampler
            )
        else:
            plot_dataloader = DataLoader(
                dataset=self._dataset, batch_size=1, shuffle=True
            )
        return plot_dataloader

    def _reset_iterator(self):
        self._sample_iterator = iter(self._dataloader)

    @staticmethod
    def _set_subplot_layout(
        num_samples: int, fig_size: tuple[int, int]
    ) -> tuple[plt.Figure, np.ndarray] | tuple[plt.Figure, plt.Axes]:
        fig, axes = plt.subplots(
            nrows=num_samples,
            ncols=1,
            sharex=True,
            sharey=False,
            figsize=fig_size,
        )

        return fig, axes

    @staticmethod
    def _set_colorbar_axes(
        fig: plt.Figure,
        coords: tuple[int, int, int, int],
        right_adjustment: float = 0.8,
    ) -> plt.Axes:
        plt.subplots_adjust(right=right_adjustment)
        return fig.add_axes(coords)

    def plot_samples(
        self,
        num_samples: int = 4,
        fig_size: tuple[int, int] = (6, 6),
        colorbar_coords: tuple[int, int, int, int] = (0.92, 0.3, 0.02, 0.4),
    ):
        fig, axes = self._set_subplot_layout(
            num_samples=num_samples, fig_size=fig_size
        )
        colorbar_axes = self._set_colorbar_axes(
            fig=fig, coords=colorbar_coords
        )

        for subplot_num in range(num_samples):
            features, label = next(self._sample_iterator)
            sns.heatmap(
                data=torch.squeeze(features),
                ax=axes[subplot_num],
                cmap="RdYlBu_r",
                cbar=(subplot_num == (num_samples - 1)),
                cbar_ax=(
                    colorbar_axes
                    if (subplot_num == (num_samples - 1))
                    else None
                ),
            )

        for i, ax in enumerate(axes):
            ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
            ax.xaxis.set_major_formatter("{x:.0f}")
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
            ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
            ax.yaxis.set_major_formatter("{x:.0f}")
            ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))
            ax2 = ax.twinx()
            ax2.tick_params(axis='y', length=0, labelsize=0,
                            labelcolor='white')
            ax2.set_ylabel("M = ?", rotation=0, labelpad=15)

        plt.show()


if __name__ == "__main__":
    my_plotter = X19MortalitySamplePlotter()
    my_plotter.plot_samples()
