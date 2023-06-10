"""Settings file"""
import unicodedata
import re
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path


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


@dataclass(frozen=True)
class AugmentationSettings:
    """Augmentation and pre-processing settings"""

    rotation_angle = 2.5
    """Rotate image in range [degrees]"""
    width_shift_range = 5.5
    """Shift image horizontally in range [%]"""
    height_shift_range = 7.5
    """Vertical shift range - [%]"""
    zoom_range = 5
    """Zoom range - [%]"""


class TrainType(Enum):
    """Model train type"""

    NORMAL = "normal"
    """No modification to dataset"""
    BALANCED = "balanced"
    """Dataset balanced"""
    WEIGHTED = "weighted"
    """Trained which class weights"""

    def __str__(self):
        return self.value


@dataclass(frozen=True)
class TrainInfo:
    """Model train information"""

    name: str
    """Model/Network name"""
    train_type: TrainType
    """Train type"""
    corruption: float
    """Dataset corruption fraction"""
    reduction: float
    """Dataset reduction fraction"""

    @property
    def report_path(self) -> Path:
        """Path to export reports like graphs"""
        _path = Path(
            "reports",
            self.safe_name,
            f"{self.train_type.value}",
            f"corruption_{self.corruption * 100:06.2f}_reduction_{self.reduction * 100:06.2f}",
        )
        _path.mkdir(parents=True, exist_ok=True)
        return _path

    @property
    def history_path(self) -> Path:
        """Path to export train history"""
        _path = Path(
            "history",
            self.safe_name,
            f"{self.train_type.value}",
            f"corruption_{self.corruption * 100:06.2f}_reduction_{self.reduction * 100:06.2f}",
        )
        _path.mkdir(parents=True, exist_ok=True)
        return _path

    @property
    def logs_path(self) -> Path:
        """Path to export logs"""
        _path = Path("logs", "models", self.safe_name, f"{self.train_type.value}")
        _path.mkdir(parents=True, exist_ok=True)
        return _path

    @property
    def safe_name(self) -> str:
        """Return safe file name to file system"""
        value = (
            unicodedata.normalize("NFKD", self.name)
            .encode("ascii", "ignore")
            .decode("ascii")
        )
        value = re.sub(r"[^\w\s-]", "", value.lower())
        return re.sub(r"[-\s]+", "-", value).strip("-_")
