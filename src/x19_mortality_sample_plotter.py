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
        weighted_random_sampling: bool = True,
        model: nn.Module = None,
    ):
        self._dataset = x19_mortality_dataset
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
        num_samples: int,
        fig_size: tuple[int, int],
        colorbar_coords: tuple[int, int, int, int],
        subplot_right_adjust: float,
    ) -> (
        tuple[plt.Figure, np.ndarray, plt.Axes]
        | tuple[plt.Figure, plt.Axes, plt.Axes]
    ):
        fig, axes = plt.subplots(
            nrows=num_samples,
            ncols=1,
            sharex=True,
            sharey=False,
            figsize=fig_size,
        )
        fig.text(0.5, 0.02, "Time (hours)", ha="center")
        fig.text(0.04, 0.5, "Feature Index", va="center", rotation="vertical")
        plt.subplots_adjust(right=subplot_right_adjust)
        colorbar_axes = fig.add_axes(colorbar_coords)

        return fig, axes, colorbar_axes

    def _label_axes(
        self, ax: plt.Axes, features: torch.tensor, actual_label: torch.tensor
    ):
        ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
        ax.xaxis.set_major_formatter("{x:.0f}")
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
        ax.yaxis.set_major_formatter("{x:.0f}")
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))
        ax2 = ax.twinx()
        ax2.tick_params(axis="y", length=0, labelsize=0, labelcolor="white")

        ax2_label = f"$M_{{act}}$ = {actual_label.item()}"

        if self._model:
            predicted_label = self._model(features)
            ax2_label = f"{ax2_label}\n$M_{{pred}}$ = {predicted_label.item()}"

        ax2.set_ylabel(
            ax2_label,
            rotation=0,
            labelpad=15,
            fontsize=9,
        )

    def plot_samples(
        self,
        num_samples: int = 4,
        fig_size: tuple[int, int] = (6, 6),
        colorbar_coords: tuple[int, int, int, int] = (0.92, 0.3, 0.02, 0.4),
        subplot_right_adjust: float = 0.8,
    ):
        fig, axes, colorbar_axes = self._set_subplot_layout(
            num_samples=num_samples,
            fig_size=fig_size,
            colorbar_coords=colorbar_coords,
            subplot_right_adjust=subplot_right_adjust,
        )

        for i, ax in enumerate(axes):
            features, class_label = next(self._sample_iterator)
            sns.heatmap(
                data=torch.squeeze(features),
                ax=ax,
                cmap="RdYlBu_r",
                cbar=(i == (axes.size - 1)),
                cbar_ax=(colorbar_axes if (i == (axes.size - 1)) else None),
            )
            self._label_axes(ax=ax, features=features, actual_label=class_label)
        plt.show()


if __name__ == "__main__":
    my_plotter = X19MortalitySamplePlotter()
    my_plotter.plot_samples()
