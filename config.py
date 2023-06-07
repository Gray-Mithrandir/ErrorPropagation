"""Settings file"""
from dataclasses import dataclass


@dataclass(frozen=True)
class PreProcessing:
    """Pre-processing settings"""

    image_size = (128, 128)
    """Image size"""
    test_images_per_class = 50
    """Number of test image used for performance testing"""


@dataclass(frozen=True)
class PlotSettings:
    """Plot Settings class"""

    plot_dpi = 300
    """Plot DPI"""
    plot_size = (10, 10)
    """Plot size in inches"""
