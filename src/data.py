"""Generates dataset for representation alignment, both "clean" and "corrupted"
such that all of the data points within the dataset share the same vocabulary.
"""

import pickle
from circuits_benchmark.benchmark.tracr_dataset import TracrDataset


def get_clean_data(case_id: int, samples: int):