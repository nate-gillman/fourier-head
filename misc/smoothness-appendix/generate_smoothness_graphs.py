import numpy as np
import matplotlib.pyplot as plt
from typing import Literal, Callable, Dict, List, Optional
from scipy.ndimage import gaussian_filter1d
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon
import collections
import math

# Set global matplotlib parameters
font_path = '../../imitation-learning/scripts/eval/Times_New_Roman.ttf'
import matplotlib.font_manager as fm
fm.fontManager.addfont(font_path)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'

import sys
sys.path.append("../..")

from smoothness_metric import (
    SIGMAS,
    get_smoothness,
    L2,
    get_l2_smoothness_measurement_function,
    get_smoothness_metric
)

class SignalGenerator:
    @staticmethod
    def make_square_wave(num_frequencies: int, x: np.ndarray, wavelength: float = 2) -> np.ndarray:
        """Generate a Fourier series approximation of a square wave.
        
        Args:
            num_frequencies: Number of frequencies to use in the approximation
            x: Input array of x values
            wavelength: The wavelength of the square wave
            
        Returns:
            Array that can be summed to get a square wave
        """
        n = np.arange(1, num_frequencies * 2, 2)
        freqs = np.broadcast_to((n * np.pi), (x.shape[0], num_frequencies))
        period = (freqs * x.reshape(-1, 1)) / wavelength
        wave = np.sin(period)
        return 4 / np.pi * (wave / n)

    @staticmethod
    def make_multinomial_square(num_sines: int, num_points: int = 2048) -> np.ndarray:
        """Create a normalized square wave using Fourier series.
        
        Args:
            num_sines: Number of sine waves to use
            num_points: Number of points in the output signal
            
        Returns:
            Normalized square wave signal
        """
        x = np.linspace(0, 10, num_points)
        square_wave = SignalGenerator.make_square_wave(num_sines, x + 21, wavelength=4).sum(axis=1)
        square_wave = np.where(square_wave < 0, 0, square_wave)
        square_wave /= square_wave.sum()
        assert math.isclose(square_wave.sum(), 1.0)
        return square_wave

class SmoothnessAnalyzer:
    def __init__(self, sigmas: Optional[np.ndarray] = None):
        """Initialize the smoothness analyzer.
        
        Args:
            sigmas: Optional array of sigma values. If None, uses default from imported module.
        """
        self.sigmas = sigmas if sigmas is not None else SIGMAS
        self.l2_smoothness = get_l2_smoothness_measurement_function(self.sigmas)
    
    def get_smoothness(self, signal: np.ndarray, D: Callable[[np.ndarray, np.ndarray], float] = L2) -> float:
        """Calculate smoothness using imported function."""
        return get_smoothness(signal, self.sigmas, D)
    
    def get_smoothness_metrics(self, signals: np.ndarray) -> Dict:
        """Calculate smoothness metrics for multiple signals."""
        return get_smoothness_metric(signals)


class Visualizer:
    @staticmethod
    def plot_fourier_square_waves(save_path: str = None):
        """Generate a 2x3 grid of Fourier square wave approximations.
        
        Args:
            save_path: Optional path to save the figure
        """

        num_sines_values = [1, 2, 3, 4, 10, 20]
        analyzer = SmoothnessAnalyzer()
        
        fig, axs = plt.subplots(3, 2, sharex=True, sharey=True, dpi=300)
        fig.set_size_inches(12, 6)
        fig.tight_layout()
        
        for ax, num_sines in zip(axs.flat, num_sines_values):
            square_wave = SignalGenerator.make_multinomial_square(num_sines)
            smoothness = analyzer.get_smoothness(square_wave, D=lambda x, y: np.linalg.norm(x - y))
            ax.plot(square_wave, color="tab:blue")
            sine_str = "Sine" if num_sines == 1 else "Sines"
            ax.set_title(f'{num_sines} {sine_str}; smoothness = {smoothness:.5f}', fontsize=12)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_smoothness_comparison(max_sines: int = 200, cutoff: int = 60, save_path: str = None):
        """Generate side-by-side comparison plot of L1 vs L2 smoothness measures.
        
        Args:
            max_sines: Maximum number of sine waves to analyze
            cutoff: Starting index for plotting results
            save_path: Optional path to save the figure
        """
        analyzer = SmoothnessAnalyzer()
        
        # Define distance metrics
        metrics = {
            "L1 Distance": lambda x, y: np.linalg.norm(x - y, ord=1),
            "L2 Distance": lambda x, y: np.linalg.norm(x - y)
        }
        
        # Calculate smoothness for each metric
        results = collections.defaultdict(list)
        for num_sines in range(1, max_sines + 1):
            square_wave = SignalGenerator.make_multinomial_square(num_sines)
            for name, metric in metrics.items():
                results[name].append(analyzer.get_smoothness(square_wave, D=metric))
        
        # Plot results
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), dpi=300)
        
        # Remove default margins
        plt.subplots_adjust(left=0.1, right=0.95, wspace=0.3)
        
        colors = ["tab:blue", "tab:orange"]
        x_range = np.arange(1, max_sines + 1)[cutoff:]
        
        # Plot L2 Distance
        ax1.plot(x_range, np.array(results["L2 Distance"])[cutoff:], 
                label="L2 Distance", color=colors[0])
        ax1.set_xlabel("Number of Sine Waves")
        ax1.legend(title="Discrepancy Measure")
        ax1.set_title('D = L2 Distance', fontsize=10)
        ax1.grid(True, which='major', linestyle='--')
        
        # Plot L1 Distance
        ax2.plot(x_range, np.array(results["L1 Distance"])[cutoff:], 
                label="L1 Distance", color=colors[1])
        ax2.set_xlabel("Number of Sine Waves")
        ax2.legend(title="Discrepancy Measure")
        ax2.set_title('D = L1 Distance', fontsize=10)
        ax2.grid(True, which='major', linestyle='--')
        
        # Add overall title
        fig.suptitle("Smoothness Metric on Square Wave with increasing Number of Sine Waves", 
                    fontsize=16, y=1.05)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

# Example usage:
if __name__ == "__main__":
    # Generate the 6 Fourier square wave plots
    Visualizer.plot_fourier_square_waves(save_path="square-wave-all-sines-combined.png")
    
    # Generate the smoothness comparison plot
    Visualizer.plot_smoothness_comparison(save_path="smoothness_graphs_l2_vs_l1.png")