"""Visualize generation metrics in a native PyQt6 window.

Usage:
    python scripts/visualize.py data/generated/users_complex_metrics.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PyQt6.QtCore import Qt, QAbstractTableModel, QModelIndex
from PyQt6.QtGui import QBrush, QColor, QFont, QPen, QPainter
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QTableView, QVBoxLayout,
    QWidget, QHeaderView, QLabel, QCheckBox, QHBoxLayout,
    QStyledItemDelegate, QStyleOptionViewItem, QAbstractItemView, QFrame,
)


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _metric_color(value: float) -> QColor:
    v = _clamp01(value)
    r = int(80 + 175 * v)
    g = int(200 - 140 * v)
    b = int(80 + 20 * v)
    return QColor(r, g, b)


def _pvalue_color(value: float) -> QColor:
    if value >= 0.05:
        return QColor(150, 220, 150)
    if value >= 0.01:
        return QColor(245, 220, 140)
    return QColor(235, 150, 150)


_KIND_COLORS = {
    "numeric": QColor(200, 220, 245),
    "integer": QColor(180, 230, 245),
    "categorical": QColor(245, 235, 190),
    "boolean": QColor(215, 240, 200),
    "identifier": QColor(225, 205, 245),
    "datetime": QColor(200, 235, 235),
    "text": QColor(245, 215, 220),
}

_UNIQ_HIGH_COLOR = QColor(180, 210, 245)
_UNIQ_ONE_COLOR = QColor(255, 100, 100)
_UNIQ_TWO_COLOR = QColor(150, 220, 150)

_IS_UNIQUE_TRUE_COLOR = QColor(150, 220, 150)
_IS_UNIQUE_FALSE_COLOR = QColor(245, 220, 140)


class StatsTableModel(QAbstractTableModel):
    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self._df = df

    def rowCount(self, parent=QModelIndex()) -> int:
        return len(self._df)

    def columnCount(self, parent=QModelIndex()) -> int:
        return len(self._df.columns)

    def _uniq_and_n(self, row: int) -> tuple[int | None, int | None]:
        try:
            uniq_val = self._df.iat[row, self._df.columns.get_loc("uniq")]
            n_val = self._df.iat[row, self._df.columns.get_loc("n")]
        except (KeyError, IndexError):
            return None, None
        try:
            uniq = int(round(float(uniq_val)))
        except (TypeError, ValueError):
            uniq = None
        try:
            n = int(round(float(n_val)))
        except (TypeError, ValueError):
            n = None
        return uniq, n

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        col_name = self._df.columns[index.column()]

        if role == Qt.ItemDataRole.DisplayRole:
            value = self._df.iat[index.row(), index.column()]
            if value is None or (isinstance(value, float) and np.isnan(value)):
                return "-"
            if col_name == "uniq":
                uniq, n = self._uniq_and_n(index.row())
                if uniq is None:
                    return "-"
                if n is not None and n > 0:
                    pct = uniq / n * 100.0
                    return f"{uniq} ({pct:.1f}%)"
                return str(uniq)
            if isinstance(value, bool):
                return "yes" if value else "no"
            if isinstance(value, float):
                return f"{value:.4f}"
            return str(value)

        if role == Qt.ItemDataRole.BackgroundRole:
            value = self._df.iat[index.row(), index.column()]
            if col_name == "kind" and isinstance(value, str):
                color = _KIND_COLORS.get(value)
                if color is not None:
                    return QBrush(color)
            if col_name == "uniq":
                uniq, n = self._uniq_and_n(index.row())
                if uniq is None:
                    return None
                if uniq == 1:
                    return QBrush(_UNIQ_ONE_COLOR)
                if uniq == 2:
                    return QBrush(_UNIQ_TWO_COLOR)
                if n is not None and uniq == n and n > 2:
                    return QBrush(_UNIQ_HIGH_COLOR)
            if col_name == "is unique":
                # Надежная проверка: значение может быть bool, str или object
                val_str = str(value).strip().lower()
                if val_str in ("true", "yes", "1"):
                    return QBrush(_IS_UNIQUE_TRUE_COLOR)
                if val_str in ("false", "no", "0"):
                    return QBrush(_IS_UNIQUE_FALSE_COLOR)
        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                return str(self._df.columns[section])
            return str(section + 1)
        if role == Qt.ItemDataRole.FontRole and orientation == Qt.Orientation.Horizontal:
            f = QFont()
            f.setBold(True)
            return f
        return None


class GroupSeparatorDelegate(QStyledItemDelegate):
    def __init__(self, parent, model: QAbstractTableModel, group_size: int):
        super().__init__(parent)
        self._model = model
        self._group_size = max(1, group_size)

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index):
        super().paint(painter, option, index)
        if self._group_size <= 0:
            return
        row = index.row()
        total = self._model.rowCount()
        if (row + 1) % self._group_size == 0 and row + 1 < total:
            painter.save()
            pen = QPen(QColor(90, 90, 90))
            pen.setWidth(2)
            painter.setPen(pen)
            rect = option.rect
            painter.drawLine(rect.left(), rect.bottom(), rect.right(), rect.bottom())
            painter.restore()


class MetricsTableModel(QAbstractTableModel):
    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self._df = df

    def rowCount(self, parent=QModelIndex()) -> int:
        return len(self._df)

    def columnCount(self, parent=QModelIndex()) -> int:
        return len(self._df.columns)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        col_name = self._df.columns[index.column()]

        if role == Qt.ItemDataRole.DisplayRole:
            value = self._df.iat[index.row(), index.column()]
            if value is None or (isinstance(value, float) and np.isnan(value)):
                return "-"
            if isinstance(value, float):
                return f"{value:.4f}"
            return str(value)

        if role == Qt.ItemDataRole.BackgroundRole:
            value = self._df.iat[index.row(), index.column()]
            if isinstance(value, float) and not np.isnan(value):
                if col_name in ("Kolmogorov-Smirnov", "Total Variation", "Jensen-Shannon"):
                    return QBrush(_metric_color(value))
                if col_name == "Kolmogorov-Smirnov p":
                    return QBrush(_pvalue_color(value))
            if col_name == "kind" and isinstance(value, str):
                color = _KIND_COLORS.get(value)
                if color is not None:
                    return QBrush(color)
        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                return str(self._df.columns[section])
            return str(section + 1)
        if role == Qt.ItemDataRole.FontRole and orientation == Qt.Orientation.Horizontal:
            f = QFont()
            f.setBold(True)
            return f
        return None


def _stats_to_dataframe(stats: list[dict]) -> pd.DataFrame:
    if not stats:
        return pd.DataFrame()
    df = pd.DataFrame(stats)
    df = df.rename(columns={
        "name": "column", "kind": "kind", "n": "n",
        "uniq": "uniq", "is_unique": "is unique",
        "mean": "mean", "std": "std", "min": "min", "max": "max",
        "top": "top", "top_share": "top share",
    })
    for col in ["uniq", "is unique"]:
        if col not in df.columns:
            df[col] = None
    # Перемещаем uniq и is unique сразу после n
    ordered = [
        "column", "kind", "n", "uniq", "is unique", "mean", "std", "min", "max",
        "top", "top share"
    ]
    return df[ordered]


def _combined_stats_dataframe(header: dict) -> pd.DataFrame:
    src = _stats_to_dataframe(header.get("source_stats", []))
    gen = _stats_to_dataframe(header.get("generated_stats", []))
    if src.empty and gen.empty:
        return pd.DataFrame()
    rows = []
    n = max(len(src), len(gen))
    for i in range(n):
        if i < len(src):
            row = src.iloc[i].copy()
            row["column"] = str(row["column"])
            rows.append(row)
        if i < len(gen):
            row = gen.iloc[i].copy()
            row["column"] = f"{row['column']} (generated)"
            rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True)


def _make_table(model, delegate=None) -> QTableView:
    table = QTableView()
    table.setModel(model)

    header = table.horizontalHeader()
    header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
    header.setStretchLastSection(False)

    table.resizeColumnsToContents()

    table.verticalHeader().setVisible(False)
    table.setAlternatingRowColors(False)
    table.setShowGrid(True)

    table.setFrameShape(QFrame.Shape.NoFrame)
    table.setSizeAdjustPolicy(QAbstractItemView.SizeAdjustPolicy.AdjustToContents)

    table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

    table.setStyleSheet(
        "QTableView { border: none; gridline-color: #c8c8c8; }"
        "QHeaderView::section { border: 1px solid #c8c8c8; padding: 2px; }"
    )

    if delegate is not None:
        table.setItemDelegate(delegate)
    return table


def _make_info_label(header: dict) -> QLabel:
    rows = header.get("generated_rows", "?")
    target = header.get("target_similarity", 0)
    achieved = header.get("achieved_similarity", 0)
    attempts = header.get("attempts_used", "?")
    source = header.get("source_file", "?")
    html = (
        f'Source: <b>{source}</b>&nbsp;&nbsp;&nbsp;'
        f'Rows: <b>{rows}</b>&nbsp;&nbsp;&nbsp;'
        f'Target similarity: <b>{target:.4f}</b>&nbsp;&nbsp;&nbsp;'
        f'Achieved: <b>{achieved:.4f}</b>&nbsp;&nbsp;&nbsp;'
        f'Attempts: <b>{attempts}</b>'
    )
    label = QLabel(html)
    label.setTextFormat(Qt.TextFormat.RichText)
    label.setContentsMargins(4, 4, 4, 4)
    return label


def make_stats_widget(header: dict) -> QWidget:
    w = QWidget()
    layout = QVBoxLayout(w)

    layout.addWidget(_make_info_label(header))

    controls = QHBoxLayout()
    controls.setContentsMargins(4, 4, 4, 4)
    combined_cb = QCheckBox("Combined View (Source + Generated)")
    controls.addWidget(combined_cb)
    controls.addStretch()
    layout.addLayout(controls)

    source_name = Path(header.get("source_file", "source")).name
    generated_name = Path(header.get("generated_csv", "generated")).name

    src_widget = QWidget()
    src_layout = QVBoxLayout(src_widget)
    src_layout.setContentsMargins(0, 0, 0, 0)
    src_title = QLabel(f"Source dataset ({source_name})")
    src_title.setContentsMargins(4, 4, 4, 4)
    src_layout.addWidget(src_title)
    src_df = _stats_to_dataframe(header.get("source_stats", []))
    src_layout.addWidget(_make_table(StatsTableModel(src_df)), alignment=Qt.AlignmentFlag.AlignLeft)

    gen_widget = QWidget()
    gen_layout = QVBoxLayout(gen_widget)
    gen_layout.setContentsMargins(0, 0, 0, 0)
    gen_title = QLabel(f"Generated dataset ({generated_name})")
    gen_title.setContentsMargins(4, 4, 4, 4)
    gen_layout.addWidget(gen_title)
    gen_df = _stats_to_dataframe(header.get("generated_stats", []))
    gen_layout.addWidget(_make_table(StatsTableModel(gen_df)), alignment=Qt.AlignmentFlag.AlignLeft)

    combined_widget = QWidget()
    combined_layout = QVBoxLayout(combined_widget)
    combined_layout.setContentsMargins(0, 0, 0, 0)
    combined_title = QLabel("Combined View (Source + Generated)")
    combined_title.setContentsMargins(4, 4, 4, 4)
    combined_layout.addWidget(combined_title)
    combined_df = _combined_stats_dataframe(header)
    combined_model = StatsTableModel(combined_df)
    combined_delegate = GroupSeparatorDelegate(combined_widget, combined_model, 2)
    combined_table = _make_table(combined_model, combined_delegate)
    combined_layout.addWidget(combined_table, alignment=Qt.AlignmentFlag.AlignLeft)

    layout.addWidget(src_widget)
    layout.addWidget(gen_widget)
    layout.addWidget(combined_widget)

    combined_widget.hide()
    layout.addStretch()

    def on_toggle(checked: bool) -> None:
        if checked:
            src_widget.hide()
            gen_widget.hide()
            combined_widget.show()
        else:
            combined_widget.hide()
            src_widget.show()
            gen_widget.show()

    combined_cb.toggled.connect(on_toggle)
    return w


def make_comparison_widget(header: dict) -> QWidget:
    w = QWidget()
    layout = QVBoxLayout(w)

    frob = header.get("frobenius_corr", 0)
    info = QLabel(f"Frobenius correlation distance: <b>{frob:.4f}</b>")
    info.setTextFormat(Qt.TextFormat.RichText)
    info.setContentsMargins(4, 4, 4, 4)
    layout.addWidget(info)

    title = QLabel("Comparison metrics (Source vs Generated)")
    title.setContentsMargins(4, 4, 4, 4)
    layout.addWidget(title)

    metrics_df = pd.DataFrame(header.get("columns", []))
    if not metrics_df.empty:
        metrics_df = metrics_df.rename(columns={
            "name": "column", "kind": "kind",
            "ks": "Kolmogorov-Smirnov", "ks_p_value": "Kolmogorov-Smirnov p",
            "tv": "Total Variation", "js": "Jensen-Shannon",
        })
        metrics_df = metrics_df[[
            "column", "kind", "Kolmogorov-Smirnov", "Kolmogorov-Smirnov p",
            "Total Variation", "Jensen-Shannon",
        ]]

    layout.addWidget(_make_table(MetricsTableModel(metrics_df)), alignment=Qt.AlignmentFlag.AlignLeft)
    layout.addStretch()
    return w


class MetricsWindow(QMainWindow):
    def __init__(self, metrics_path: Path):
        super().__init__()
        self.setWindowTitle(f"Metrics: {metrics_path.name}")
        self.resize(1250, 800)

        header = json.loads(metrics_path.read_text(encoding="utf-8"))

        tabs = QTabWidget()
        tabs.addTab(make_stats_widget(header), "Data metrics")
        tabs.addTab(make_comparison_widget(header), "Comparison")
        self.setCentralWidget(tabs)


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python scripts/visualize.py <metrics.json>", file=sys.stderr)
        return 2
    path = Path(sys.argv[1])
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        return 2

    app = QApplication(sys.argv)
    win = MetricsWindow(path)
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())