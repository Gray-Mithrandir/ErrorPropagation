"""Model train tracker
Track on train point and allow to skip already trained
"""
from __future__ import annotations

import json
from collections.abc import MutableMapping
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Dict


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
    """Status of train point"""

    reduction: int
    """Reduction percent"""
    corruption: int
    """Corruption percent"""
    train_type: TrainType
    """Train type"""

    def __post_init__(self):
        if not isinstance(self.reduction, int):
            object.__setattr__(self, "reduction", int(self.reduction))
        if not isinstance(self.corruption, int):
            object.__setattr__(self, "corruption", int(self.corruption))

    @classmethod
    def from_string(cls, string: str) -> TrainInfo:
        """Create class from string representation
        String format should be - <train_type.value>;<reduction>;<corruption>

        Parameters
        ----------
        string: str
            String to parse

        Returns
        -------
        TrainInfo
            Parsed class
        """
        _split = string.split(";")
        return TrainInfo(
            train_type=TrainType(_split[0]),
            reduction=int(_split[1]),
            corruption=int(_split[2]),
        )

    def to_string(self) -> str:
        """Create string representation
        String format - <train_type.value>;<reduction>;<corruption>
        """
        return f"{self.train_type.value};{self.reduction:.0f};{self.corruption:.0f}"


@dataclass
class TrainStatus:
    """Train status"""

    train: bool = False
    """Set if model fit completed and all history exported"""
    evaluation: bool = False
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
        self._points = {}  # type: Dict[TrainInfo, TrainStatus]
        if not self._path.parent.exists():
            self._path.parent.mkdir(parents=True)
        if self._path.exists():
            with open(self._path, mode="r", encoding="utf-8") as json_fh:
                import_dict = json.load(json_fh)
                for str_info, dict_status in import_dict.items():
                    info = TrainInfo.from_string(str_info)
                    status = TrainStatus(
                        train=dict_status.get("train", False),
                        evaluation=dict_status.get("evaluation", False),
                    )
                    self._points[info] = status

    def set_train_complete(self, train_info: TrainInfo):
        """Mark point as train process completed

        Parameters
        ----------
        train_info: TrainInfo
            Train information
        """
        _status = self._points.get(train_info, TrainStatus())
        _status.train = True
        self._points[train_info] = _status
        self._save()

    def set_evaluation_complete(self, train_info: TrainInfo):
        """Mark point as evaluation process completed

        Parameters
        ----------
        train_info: TrainInfo
            Train information

        Raises
        ------
        ValueError
            If train status is false
        """
        _status = self._points.get(train_info, TrainStatus())
        if not _status.train:
            raise ValueError("Can't set evaluation done without train")
        _status.evaluation = True
        self._points[train_info] = _status
        self._save()

    def is_point_trained(self, train_info: TrainInfo) -> bool:
        """Check if train process already competed

        Parameters
        ----------
        train_info: TrainInfo
            Train information
        """
        return self._points.get(train_info, TrainStatus).train

    def is_point_evaluated(self, train_info: TrainInfo) -> bool:
        """Check if evaluation process already competed

        Parameters
        ----------
        train_info: TrainInfo
            Train information
        """
        return self._points.get(train_info, TrainStatus).evaluation

    def __getitem__(self, item):
        return self._points.get(item, TrainStatus())

    def __setitem__(self, key, value):
        self._points[key] = value
        self._save()

    def __delitem__(self, key):
        del self._points[key]
        self._save()

    def __iter__(self):
        yield from self._points

    def __len__(self):
        return len(self._points)

    def _save(self):
        export_dict = {
            point.to_string(): asdict(status) for point, status in self._points.items()
        }
        with open(self._path, mode="w", encoding="utf-8") as json_fh:
            json.dump(export_dict, json_fh, indent=4, sort_keys=True)
