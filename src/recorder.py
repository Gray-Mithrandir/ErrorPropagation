"""Record training statistics"""
import logging
from pathlib import Path
from typing import Tuple

import openpyxl
import tensorflow as tf
from openpyxl.styles import Alignment
from openpyxl.utils.cell import get_column_letter
from openpyxl.worksheet.worksheet import Worksheet
from contextlib import contextmanager

from tracker import TrainType, TrainPoint
from load_dataset import get_test_dataset, create_class_filter


class TrainStatistics:
    """Record evaluation results"""

    def __init__(self, export_path: Path):
        """Create or load statistic file

        Parameters
        ----------
        export_path: Path
            Folder to save excel report
        """
        self.report_path = Path("reports")
        self.logger = logging.getLogger("raido.report")
        self.report_path = (export_path / "statistics.xlsx").absolute()
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.report_path.exists():
            self.logger.info("Workbook not exist. Creating a new")
            self._create_new()

    def save_train_history(
        self, train_info: TrainPoint, history: tf.keras.callbacks.History
    ) -> None:
        """Record model training information and create plots

        Parameters
        ----------
        train_info: TrainPoint
            Model training information
        history: tf.keras.callbacks.History
            Train history
        """
        self.logger.info("Recording new train metrics for %s", train_info)
        with self._get_workbook() as workbook:
            sheets = (
                workbook["Accuracy"],
                workbook["Loss"],
                workbook["ValidationAccuracy"],
                workbook["ValidationLoss"],
            )
            metrics = ("accuracy", "loss", "val_accuracy", "val_loss")
            for sheet, metric_name in zip(sheets, metrics):
                row = self._get_metric_cell(sheet, train_info)
                sheet[self._get_metric_cell(sheet, train_info)] = history.history[
                    metric_name
                ][-1]

    def save_evaluation(self, model: tf.keras.Model, train_info: TrainPoint) -> None:
        """Save evaluation metrics on test dataset

        Parameters
        ----------
        model: tf.keras.Model
            Model to evaluate
        train_info: TrainPoint
            Train information
        """
        test_ds = get_test_dataset()
        self.logger.info("Recording evaluation metrics for %s", train_info)
        with self._get_workbook() as workbook:
            loss_cells = self._get_class_cell(workbook["LossPerClass"], train_info)
            acc_cells = self._get_class_cell(workbook["AccuracyPerClass"], train_info)
            for class_index, (a_cell, l_cell) in enumerate(zip(acc_cells, loss_cells)):
                _class_ds = test_ds.filter(create_class_filter(class_index)).batch(1)
                loss, accuracy = model.evaluate(_class_ds)
                workbook["AccuracyPerClass"][a_cell] = accuracy
                workbook["LossPerClass"][l_cell] = loss

    @contextmanager
    def _get_workbook(self) -> openpyxl.Workbook:
        """Create or open XLS file

        Returns
        -------
        openpyxl.Workbook
            New or existing workbook
        """
        self.logger.info("Loading workbook")
        workbook = openpyxl.load_workbook(filename=self.report_path, data_only=True)
        yield workbook
        self.logger.info("Saving workbook")
        workbook.save(self.report_path)

    def _create_new(self) -> None:
        """Create new workbook"""
        classes_names = get_test_dataset().class_names
        new_wb = openpyxl.Workbook()
        new_wb.create_sheet("Accuracy")
        new_wb.create_sheet("Loss")
        new_wb.create_sheet("ValidationAccuracy")
        new_wb.create_sheet("ValidationLoss")
        for sheet in (
            new_wb["Accuracy"],
            new_wb["Loss"],
            new_wb["ValidationAccuracy"],
            new_wb["ValidationLoss"],
        ):
            sheet["A1"] = "Reduction"
            sheet["B1"] = "Corruption"
            for offset, train_type in enumerate(TrainType, start=3):
                sheet[f"{get_column_letter(offset)}1"] = f"{train_type}".capitalize()
        new_wb.create_sheet("AccuracyPerClass")
        new_wb.create_sheet("LossPerClass")
        for sheet in (new_wb["AccuracyPerClass"], new_wb["LossPerClass"]):
            sheet.merge_cells("A1:A2")
            cell = sheet.cell(row=1, column=1)
            cell.value = "Reduction"
            cell.alignment = Alignment(horizontal="center", vertical="center")
            sheet.merge_cells("B1:B2")
            cell = sheet.cell(row=1, column=2)
            cell.value = "Corruption"
            cell.alignment = Alignment(horizontal="center", vertical="center")
            for col_offset, train_type in enumerate(TrainType):
                cell_index = len(classes_names) * col_offset + 3
                sheet.merge_cells(
                    f"{get_column_letter(cell_index)}1:{get_column_letter(cell_index + len(classes_names) - 1)}1"
                )
                cell = sheet.cell(row=1, column=cell_index)
                cell.value = f"{train_type}".capitalize()
                cell.alignment = Alignment(horizontal="center", vertical="center")
                for class_offset, class_name in enumerate(classes_names):
                    sheet[
                        f"{get_column_letter(cell_index + class_offset)}2"
                    ] = f"{class_name}".capitalize()
        del new_wb["Sheet"]
        new_wb.save(self.report_path)

    def _get_metric_cell(self, sheet: Worksheet, train_info: TrainPoint) -> str:
        """Find first empty point or return existing one according to train_info

        Parameters
        ----------
        sheet: Worksheet
            Worksheet to search
        train_info: TrainPoint
            Train information

        Returns
        -------
        str
            First empty or matched cell
        """
        offset_col = 2 + list(_type for _type in TrainType).index(train_info.train_type)
        row_index = 2
        for row_index, row in enumerate(
            sheet.iter_rows(min_row=row_index, min_col=1, max_col=5), start=row_index
        ):
            if row[0].value == train_info.reduction and row[1].value == train_info.corruption:
                self.logger.debug(
                    "Cell found at coordinate %s", row[offset_col].coordinate
                )
                return row[offset_col].coordinate
        sheet[f"A{row_index+1}"] = train_info.reduction
        sheet[f"B{row_index+1}"] = train_info.corruption
        cell_coord = f"{get_column_letter(offset_col)}{row_index+1}"
        self.logger.info("Cell not found.Appending %s", cell_coord)
        return cell_coord

    def _get_class_cell(
        self, sheet: Worksheet, train_info: TrainPoint
    ) -> Tuple[str, str, str]:
        """Find first empty point or return existing one according to train_info

        Parameters
        ----------
        sheet: Worksheet
            Worksheet to search
        train_info: TrainPoint
            Train information

        Returns
        -------
        Tuple[str, str, str]
            First empty or matched cells (for all classes)
        """
        offset_col = (
            2 + list(_type for _type in TrainType).index(train_info.train_type) * 3
        )
        row_index = 3
        for row_index, row in enumerate(sheet.iter_rows(min_row=row_index, min_col=1), start=row_index):
            self.logger.debug(
                "Testing cell at coordinate %s", row[offset_col].coordinate
            )
            if row[0].value == train_info.reduction and row[1].value == train_info.corruption:
                self.logger.debug(
                    "Cell found at coordinate %s", row[offset_col].coordinate
                )
                return (
                    row[offset_col].coordinate,
                    row[offset_col + 1].coordinate,
                    row[offset_col + 2].coordinate,
                )
        sheet[f"A{row_index+1}"] = train_info.reduction
        sheet[f"B{row_index+1}"] = train_info.corruption
        cell_coord = (
            f"{get_column_letter(offset_col+1)}{row_index+1}",
            f"{get_column_letter(offset_col+2)}{row_index+1}",
            f"{get_column_letter(offset_col+3)}{row_index+1}",
        )
        self.logger.info("Cell not found.Appending %s", cell_coord)
        return cell_coord
