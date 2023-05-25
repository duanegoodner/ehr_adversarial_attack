from enum import Enum, auto
from data_structures import EvalEpochResult


class OptimizeDirection(Enum):
    MIN = auto()
    MAX = auto()


class EarlyStopping:
    def __init__(
        self,
        performance_metric: str,
        optimize_direction: OptimizeDirection,
        patience: int,
    ):
        self._optimize_direction = optimize_direction
        self._performance_metric = performance_metric
        self._patience = patience
        self._num_deteriorations = 0
        if optimize_direction == OptimizeDirection.MIN:
            self._best_result = float("inf")
        if optimize_direction == OptimizeDirection.MAX:
            self._best_result = float("max")

    def _is_greater_than_best_result(self, result: EvalEpochResult) -> bool:
        return getattr(result, self._performance_metric) > self._best_result

    def _is_less_than_best_result(self, result: EvalEpochResult) -> bool:
        return getattr(result, self._performance_metric) < self._best_result

    def _reset_num_deteriorations(self):
        self._num_deteriorations = 0

    def _is_worse_than_best_result(self, result: EvalEpochResult) -> bool:

        dispatch_table = {
            OptimizeDirection.MIN: self._is_greater_than_best_result,
            OptimizeDirection.MAX: self._is_less_than_best_result
        }

        return dispatch_table[self._optimize_direction](result)

    def _check_and_update(self, result: EvalEpochResult):
        if self._is_worse_than_best_result(result=result):
            self._num_deteriorations += 1
        else:
            self._reset_num_deteriorations()

    def indicates_early_stop(self, result: EvalEpochResult):
        self._check_and_update(result=result)
        return self._num_deteriorations > self._patience




