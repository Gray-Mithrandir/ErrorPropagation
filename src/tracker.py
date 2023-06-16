"""Model train tracker
Track on train point and allow to skip already trained
"""
from dataclasses import dataclass, asdict
import json
from enum import Enum
from pathlib import Path
from typing import Dict, Union
from collections.abc import MutableMapping


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
class TrainPoint:
    """Status of train point"""

    reduction: float
    """Reduction faction"""
    corruption: float
    """Corruption faction"""
    train_type: TrainType
    """Train type"""

    def __post_init__(self):
        if not (0 <= self.reduction <= 1):
            raise ValueError("Reduction must be in range [0-1]")
        if not (0 <= self.corruption <= 1):
            raise ValueError("Corruption must be in range [0-1]")

    def dict(self) -> Dict[str, Union[str, TrainType]]:
        """Return dict representation"""
        return asdict(self)


@dataclass
class TrainStatus:
    """Train status"""

    train_complete: bool = False
    """Set if model fit completed and all history exported"""
    evaluation_exported: bool = False
    """Set if model evaluation metrics was saved"""


class RunTracker(MutableMapping):
    """Track model train process"""

    def __init__(self, tracker_path: Path):
        """Initialize new or load existing tracker

        Parameters
        ----------
        tracker_path: Path
            File name to save track info
        """
        self._path = tracker_path
        self._points = {}  # type: Dict[TrainPoint, TrainStatus]
        if not self._path.parent.exists():
            self._path.parent.mkdir(parents=True)
        if self._path.exists():
            with open(self._path, mode="r", encoding="utf-8") as json_fh:
                import_dict = json.load(json_fh)
                for point, value in import_dict.items():
                    point_split = point.split(";")
                    point_obj = TrainPoint(
                        reduction=float(point_split[0].replace("r", "")),
                        corruption=float(point_split[1].replace("c", "")),
                        train_type=TrainType(point_split[2].replace("t", "")),
                    )
                    self._points[point_obj] = TrainStatus(**value)

    def mark_train_complete(
        self, reduction: float, corruption: float, train_type: TrainType
    ):
        """Mark point as train process completed

        Parameters
        ----------
        reduction: float
            Reduction fraction
        corruption: float
            Corruption faction
        train_type: TrainType
            Model train type
        """
        _point = TrainPoint(
            reduction=reduction, corruption=corruption, train_type=train_type
        )
        _status = self._points.get(_point, TrainStatus())
        _status.train_complete = True
        self._points[_point] = _status
        self._save()

    def mark_evaluation_complete(
        self, reduction: float, corruption: float, train_type: TrainType
    ):
        """Mark point as evaluation process completed

        Parameters
        ----------
        reduction: float
            Reduction fraction
        corruption: float
            Corruption faction
        train_type: TrainType
            Model train type
        """
        _point = TrainPoint(
            reduction=reduction, corruption=corruption, train_type=train_type
        )
        _status = self._points.get(_point, TrainStatus())
        if not _status.train_complete:
            raise ValueError("Can't set evaluation complete without train complete")
        _status.evaluation_exported = True
        self._points[_point] = _status
        self._save()

    def is_point_trained(
        self, reduction: float, corruption: float, train_type: TrainType
    ) -> bool:
        """Check if train process already competed

        Parameters
        ----------
        reduction: float
            Reduction fraction
        corruption: float
            Corruption faction
        train_type: TrainType
            Model train type
        """
        _point = TrainPoint(
            reduction=reduction, corruption=corruption, train_type=train_type
        )
        return _point in self._points and self._points[_point].train_complete

    def is_point_evaluated(
        self, reduction: float, corruption: float, train_type: TrainType
    ) -> bool:
        """Check if evaluation process already competed

        Parameters
        ----------
        reduction: float
            Reduction fraction
        corruption: float
            Corruption faction
        train_type: TrainType
            Model train type
        """
        _point = TrainPoint(
            reduction=reduction, corruption=corruption, train_type=train_type
        )
        return _point in self._points and self._points[_point].evaluation_exported

    def __getitem__(self, item):
        if not isinstance(item, TrainPoint):
            return NotImplemented
        return self._points.get(item, TrainStatus())

    def __setitem__(self, key, value):
        if not (isinstance(key, TrainPoint) and isinstance(value, TrainStatus)):
            return NotImplemented
        self._points[key] = value
        self._save()

    def __delitem__(self, key):
        if not isinstance(key, TrainPoint):
            return NotImplemented
        del self._points[key]
        self._save()

    def __iter__(self):
        yield from self._points

    def __len__(self):
        return len(self._points)

    def _save(self):
        export_dict = {
            f"r{point.reduction};c{point.corruption};t{point.train_type.value}": asdict(
                status
            )
            for point, status in self._points.items()
        }
        with open(self._path, mode="w", encoding="utf-8") as json_fh:
            json.dump(export_dict, json_fh, indent=4)
