"""
This is the metrics' module.

This module contains and expose the recommendation metrics.
Each metric is encapsulated in a specific package.

See the implementation of Precision metric for creating new per-user metrics.
See the implementation of Item Coverage for creating new cross-user metrics.
"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

from elliot.evaluation.metrics.accuracy.ndcg import nDCG
from elliot.evaluation.metrics.bias import Calibration, EnergyCalibration
from elliot.evaluation.metrics.statistical_array_metric import StatisticalMetric

_metric_dictionary = {
    "nDCG": nDCG,
    "Calibration": Calibration,
    "EnergyCalibration": EnergyCalibration,
}
_lower_dict = {k.lower(): v for k, v in _metric_dictionary.items()}


def parse_metrics(metrics):
    return [_lower_dict[m.lower()] for m in metrics if m.lower() in _lower_dict.keys()]


def parse_metric(metric):
    metric = metric.lower()
    return _lower_dict[metric] if metric in _lower_dict.keys() else None
