import numpy as np
from scipy.stats import norm, beta


def generate_gaussian_dataset(n_samples, var=0.1, seed=42):
    """
    Generates a 3D dataset with n_samples samples.

    The dataset is generated as follows:
    1. x is sampled uniformly from (-0.8, 0.8)
    2. y is sampled from a Gaussian centered at x with variance var
    3. z is sampled from a Gaussian centered at y with variance var

    Parameters:
    - n_samples (int): Number of samples to generate.

    Returns:
    - dataset (ndarray): An array of shape (n_samples, 3) containing the 3D dataset.
    """
    rng = np.random.default_rng(seed=seed)
    # Step 1: Sample x uniformly from (-0.8, 0.8)
    x = rng.uniform(-0.8, 0.8, n_samples)

    # Step 2: Sample y from a Gaussian centered at x with variance var
    y = rng.normal(loc=x, scale=np.sqrt(var))

    # Step 3: Sample z from a Gaussian centered at y with variance var
    z = rng.normal(loc=y, scale=np.sqrt(var))

    # Combine x, y, z into a single dataset
    dataset = np.vstack((x, y, z)).T
    return dataset

def gaussian_pdf(bin_centers, loc, var=0.01):
    mean = loc[0]
    if len(loc) == 2:
        mean = loc[1]
    pmf =  norm.pdf(bin_centers, mean, np.sqrt(var))*2 / bin_centers.shape[0]
    return pmf / np.sum(pmf)

def generate_gmm_dataset(n_samples, var=0.01, seed=42):
    """
    Generates a 3D dataset with n_samples samples.

    The dataset is generated as follows:
    1. x is sampled uniformly from (-0.8, 0.8)
    2. y is sampled from a Gaussian centered at x with variance 0.01
    3. z is sampled from a GMM with means min{x,y}-0.1 and max{x,y}+0.1, each with variance 0.01

    Parameters:
    - n_samples (int): Number of samples to generate.

    Returns:
    - dataset (ndarray): An array of shape (n_samples, 3) containing the 3D dataset.
    """

    rng = np.random.default_rng(seed=seed)
    # Step 1: Sample x uniformly from (-0.8, 0.8)
    x = rng.uniform(-0.8, 0.8, n_samples)

    # Step 2: Sample y from a Gaussian centered at x with variance 0.01
    y = rng.normal(loc=x, scale=np.sqrt(var), size=n_samples)

    # Step 3: Sample z from a GMM with means x and y, each with variance 0.01
    z = np.zeros(n_samples)

    a = np.minimum(x,y) - 0.1
    b = np.maximum(x,y) + 0.1
    for i in range(n_samples):
        # Randomly choose either x[i] or y[i] as the mean for z
        if rng.uniform(0, 1) < 0.5:
            z[i] = rng.normal(loc=a[i], scale=np.sqrt(var))
        else:
            z[i] = rng.normal(loc=b[i], scale=np.sqrt(var))

    # Combine x, y, z into a single dataset
    dataset = np.vstack((x, y, z)).T
    return dataset

def generate_gmm_dataset2(n_samples, var=0.01, seed=42):
    """
    Generates a 3D dataset with n_samples samples.

    The dataset is generated as follows:
    1. x and y are sampled uniformly from (-0.8, 0.8)
    3. z is sampled from a GMM with means x and y, each with variance var

    Parameters:
    - n_samples (int): Number of samples to generate.

    Returns:
    - dataset (ndarray): An array of shape (n_samples, 3) containing the 3D dataset.
    """
    rng = np.random.default_rng(seed=seed)
    # Step 1: Sample x uniformly from (-0.8, 0.8)
    x = rng.uniform(-0.8, 0.8, n_samples)
    y = rng.uniform(-0.8, 0.8, n_samples)

    # Step 3: Sample z from a GMM with means x and y, each with variance 0.01
    z = np.zeros(n_samples)
    for i in range(n_samples):
        # Randomly choose either x[i] or y[i] as the mean for z
        if rng.uniform(0, 1) < 0.5:
            z[i] = rng.normal(loc=x[i], scale=np.sqrt(var))
        else:
            z[i] = rng.normal(loc=y[i], scale=np.sqrt(var))

    # Combine x, y, z into a single dataset
    dataset = np.vstack((x, y, z)).T
    return dataset


def gmm1_pdf(bin_centers, locs, var=0.01):
    return (gaussian_pdf(bin_centers, [np.min(locs)-0.1], var) + gaussian_pdf(bin_centers, [np.max(locs)+0.1], var))/2

def gmm2_pdf(bin_centers, locs, var=0.01):
    return (gaussian_pdf(bin_centers, [locs[0]], var) + gaussian_pdf(bin_centers, [locs[1]], var))/2


def generate_beta_dataset(n_samples, var=0.01, seed=42):
    """
    Generates a 3D dataset with n_samples samples.

    The dataset is generated as follows:
    1. x is sampled uniformly from (-0.8, 0.8)
    2. y is sampled from a Gaussian centered at x with variance var
    3. z is sampled from Random({1, -1}) x Beta(100|x|, 100|y|)

    Parameters:
    - n_samples (int): Number of samples to generate.

    Returns:
    - dataset (ndarray): An array of shape (n_samples, 3) containing the 3D dataset.
    """
    rng = np.random.default_rng(seed=seed)
    # Step 1: Sample x uniformly from (-0.8, 0.8)
    x = rng.uniform(-0.8, 0.8, n_samples)

    # Step 2: Sample y from a Gaussian centered at x with variance var
    y = rng.normal(loc=x, scale=np.sqrt(var))

    # Step 3: Sample z from Beta
    sign = rng.choice([1, -1], size=n_samples)
    z =  np.array([sign[i] * rng.beta(np.abs(100*x[i]), np.abs(100*y[i])) for i in range(n_samples)])
    
    # Combine x, y, z into a single dataset
    dataset = np.vstack((x, y, z)).T
    return dataset

def beta_pdf(bin_centers, locs, var=None):
    pos = bin_centers[bin_centers >= 0]
    pmf =  beta.pdf(pos, np.abs(100*locs[0]), np.abs(100*locs[1])) * 1 / (2 * pos.shape[0])
    pmf = np.concatenate((pmf[::-1], pmf))
    return pmf / np.sum(pmf)
