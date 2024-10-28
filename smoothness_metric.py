"""
The MIT License (MIT) Copyright (c) 2024 Nate Gillman

Permission is hereby granted, free of charge, to any person obtaining a copy of this software 
and associated documentation files (the "Software"), to deal in the Software without restriction, 
including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, 
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial 
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT 
LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. 
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE 
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from typing import Callable, Optional
from functools import partial
from scipy.ndimage import gaussian_filter1d
from scipy.stats import entropy
import numpy as np
import pandas as pd
from tqdm import tqdm

MAX_SIGMA = 100
SIGMAS = np.arange(1, MAX_SIGMA + 1)

def _get_smoothness_single_sigma(multinomial, stddev, D: Callable[[np.ndarray, np.ndarray], float]) -> float:
    radius = multinomial.shape[0]-1
    filtered_multinomial = gaussian_filter1d(multinomial, stddev, mode="wrap",radius=radius) # wrapping padding
    return D(multinomial, filtered_multinomial)

def _get_alpha(sigma) -> float:
    # notice that   \sum_{sigma=1}^\infty \frac{1}{sigma^2} = \pi^2/6
    # thus,         \sum_{sigma=1}^\infty _get_alpha(sigma) = 1
    return (6 / (np.pi ** 2)) * (1/ sigma ** 2)

def _get_weighted_smoothness(multinomial, stddev, D):
    return _get_alpha(stddev) * _get_smoothness_single_sigma(multinomial, stddev, D)

def get_smoothness(multinomial, sigmas, D):
    return np.array([_get_weighted_smoothness(multinomial, sigma, D) for sigma in sigmas]).sum()

# define some notions of 'distance' or divergence
L2 = lambda x, y: np.linalg.norm(x - y)

metric_funcs = {
    "L2": lambda x: get_smoothness(x, sigmas=SIGMAS, D=L2),
}

def get_l2_smoothness_measurement_function(sigmas: Optional[np.ndarray] = None) -> Callable[[np.ndarray], float]:
    return metric_funcs["L2"]

def get_smoothness_metric(arr, metric_funcs=metric_funcs):
    total_distributions = arr.shape[0]

    results = []
    for k in tqdm(range(total_distributions)):
        m = arr[k,:]
        for metric_name, metric_func in metric_funcs.items():
            results.append({
                "metric": metric_name,
                "value": metric_func(m),
            })

    df = pd.json_normalize(results)
    smoothness_summary = df.groupby(by=["metric"]).describe()

    results_per_metric = {}
    for metric_name, metric_func in metric_funcs.items():
        mean = smoothness_summary.loc[metric_name, ("value", "mean")]
        std = smoothness_summary.loc[metric_name, ("value", "std")]
        results_per_metric[metric_name] = {"mean" : mean, "std" : std}

    return results_per_metric