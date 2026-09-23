"""
Renders a flat word-level OCR DataFrame (columns: `text`, `boundingBox` =
[xmin, ymin, xmax, ymax]) directly into an HTML <table>, preserving the
source document's visual layout — including automatic colspan/rowspan
wherever a word's bounding box actually spans more than one grid cell.

The approach: cluster words into visual rows by vertical bounding-box
overlap (`group_words_into_rows`), lay out a uniform grid sized from the
average word width/height, then merge whichever grid cells each word's
bounding box actually overlaps (so a wide header cell renders as one merged
cell instead of being force-split into narrower columns), collapse empty
cell runs, and render the result to HTML. Because it starts from the same
flat `(text, boundingBox)` DataFrame contract every recognizer in this
project produces, it works unchanged regardless of which OCR backend
(`PaddleTextRecognizer`, `AzureTextRecognizer`, or a custom one satisfying
`recognizers.protocols.TextRecognizerProtocol`) generated the words.
"""
import html as html_lib
import math
from typing import List, Tuple

import pandas as pd

# Grid cells wider than this are almost always a bad cell-size estimate
# (e.g. one stray tiny word skewing the mean width down) rather than a
# real wide table — cap it so a bad estimate can't blow up the grid.
MAX_GRID_COLUMNS = 200


def _vertical_overlap_pct(interval1: Tuple[float, float], interval2: Tuple[float, float]) -> float:
    y1_min, y1_max = interval1
    y2_min, y2_max = interval2
    overlap = max(0.0, min(y1_max, y2_max) - max(y1_min, y2_min))
    total = max(y1_max, y2_max) - min(y1_min, y2_min)
    return (overlap / total * 100) if total > 0 else 0.0


def group_words_into_rows(df_word: pd.DataFrame, overlap_threshold: float = 50.0) -> List[List[Tuple[str, List[int]]]]:
    """
    Cluster OCR words into visual rows (lines) by vertical bounding-box
    overlap against the row-so-far, independent of which OCR backend
    produced them. `df_word` is expected sorted top-to-bottom then
    left-to-right (both bundled recognizers already sort their output
    this way).

    Returns a list of rows, each row a list of (text, bbox) tuples sorted
    left-to-right.
    """
    if df_word is None or df_word.empty:
        return []

    words = sorted(
        df_word[['text', 'boundingBox']].itertuples(index=False, name=None),
        key=lambda w: (w[1][1], w[1][0]),
    )

    rows: List[List[Tuple[str, List[int]]]] = [[words[0]]]
    for word in words[1:]:
        bbox = word[1]
        row = rows[-1]
        row_y1 = min(w[1][1] for w in row)
        row_y2 = max(w[1][3] for w in row)
        if _vertical_overlap_pct((bbox[1], bbox[3]), (row_y1, row_y2)) >= overlap_threshold:
            row.append(word)
        else:
            rows.append([word])

    for row in rows:
        row.sort(key=lambda w: w[1][0])

    return rows


def compute_table_dimensions(df_word: pd.DataFrame, cell_height: float, cell_width: float):
    """
    Estimate how many grid rows/columns are needed to cover all detected
    words, given a fixed cell size.
    """
    if df_word is None or df_word.empty or cell_width <= 0 or cell_height <= 0:
        return 0, 0, 0, 0

    min_y = min(bbox[1] for bbox in df_word['boundingBox'])
    max_x = max(bbox[2] for bbox in df_word['boundingBox'])
    max_y = max(bbox[3] for bbox in df_word['boundingBox'])

    total_width = max_x
    total_height = max_y - min_y

    num_cols = math.ceil(total_width / cell_width)
    num_rows = math.ceil(total_height / cell_height)
    return num_rows, num_cols, max_x, max_y


class Cell:
    def __init__(self, row: int, col: int, start_x: float, start_y: float, content: str, height: float, width: float) -> None:
        self.row = row
        self.col = col
        self.content = content
        self.start_x = start_x
        self.start_y = start_y
        self.col_span = 1
        self.row_span = 1
        self.width = width
        self.height = height

    def __repr__(self) -> str:
        return f"Cell(row={self.row}, col={self.col}, content={self.content!r}, col_span={self.col_span})"


class Grid:
    """A fixed-size grid of Cells that words get merged into by bounding-box overlap."""

    MERGE_OVERLAP_THRESHOLD = 20

    def __init__(self, total_rows: int, total_cols: int, cell_width: float, cell_height: float) -> None:
        self.rows: List[List[Cell]] = [
            [
                Cell(r, c, c * cell_width, r * cell_height, '', cell_height, cell_width)
                for c in range(total_cols)
            ]
            for r in range(total_rows)
        ]

    @staticmethod
    def _overlap_pct(rect1: List[float], rect2: List[float]) -> float:
        x_left = max(rect1[0], rect2[0])
        y_top = max(rect1[1], rect2[1])
        x_right = min(rect1[2], rect2[2])
        y_bottom = min(rect1[3], rect2[3])

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        rect1_area = (rect1[2] - rect1[0]) * (rect1[3] - rect1[1])
        rect2_area = (rect2[2] - rect2[0]) * (rect2[3] - rect2[1])
        min_area = min(rect1_area, rect2_area)
        return round((intersection_area / min_area) * 100, 2) if min_area != 0 else 0.0

    def _merge_cells(self, row_index: int, cell_indices: List[int], new_content: str) -> None:
        """Fold every cell in cell_indices into one (the first), appending new_content to it."""
        if not cell_indices:
            return
        new_row: List[Cell] = []
        merged = 0
        updated_cell = None
        for ind, cell in enumerate(self.rows[row_index]):
            if ind in cell_indices:
                if merged == 0:
                    updated_cell = cell
                else:
                    updated_cell.content = f"{updated_cell.content} {cell.content}".strip()
                    updated_cell.col_span += 1
                    updated_cell.width += cell.width
                merged += 1
                if merged == len(cell_indices):
                    updated_cell.content = f"{updated_cell.content} {new_content}".strip()
                    new_row.append(updated_cell)
            else:
                new_row.append(cell)
        self.rows[row_index] = new_row

    def build(self, rows_of_words: List[List[Tuple[str, List[int]]]], merge_close_words: bool = False) -> None:
        for row_index, row_words in enumerate(rows_of_words):
            if row_index >= len(self.rows):
                break
            for text, bbox in row_words:
                overlapping_indices: List[int] = []
                entered = False
                for cell_index, cell in enumerate(self.rows[row_index]):
                    cell_bbox = [cell.start_x, cell.start_y, cell.start_x + cell.width, cell.start_y + cell.height]
                    word_bbox_adj = [bbox[0], cell_bbox[1], bbox[2], cell_bbox[3]]
                    overlap = self._overlap_pct(word_bbox_adj, cell_bbox)
                    if overlap >= self.MERGE_OVERLAP_THRESHOLD:
                        entered = True
                        overlapping_indices.append(cell_index)
                    elif entered:
                        break
                if overlapping_indices:
                    self._merge_cells(row_index, overlapping_indices, text)

        self._postprocess(merge_close_words=merge_close_words)

    def _postprocess(self, merge_close_words: bool) -> None:
        # Always collapse consecutive empty cells into one wider empty cell.
        for row_ind, row in enumerate(self.rows):
            new_cells: List[Cell] = []
            start = 0
            update_cell = None
            for cell in row:
                if cell.content == '' and start == 0:
                    start += 1
                    update_cell = cell
                elif cell.content == '' and start > 0:
                    start += 1
                    update_cell.col_span += cell.col_span
                    update_cell.width += cell.width
                else:
                    if start > 0:
                        new_cells.append(update_cell)
                    new_cells.append(cell)
                    start = 0
            if start > 0:
                new_cells.append(update_cell)
            self.rows[row_ind] = new_cells

        if merge_close_words:
            # Optionally also merge consecutive non-empty cells together
            # (useful when a value got split across more grid cells than
            # it should have).
            for row_ind, row in enumerate(self.rows):
                new_cells = []
                start = 0
                update_cell = None
                for cell in row:
                    if cell.content.strip() != '' and start == 0:
                        start += 1
                        update_cell = cell
                    elif cell.content.strip() != '' and start > 0:
                        start += 1
                        update_cell.col_span += cell.col_span
                        update_cell.content += ' ' + cell.content
                        update_cell.width += cell.width
                    else:
                        if start > 0:
                            new_cells.append(update_cell)
                        new_cells.append(cell)
                        start = 0
                if start > 0:
                    new_cells.append(update_cell)
                self.rows[row_ind] = new_cells

    def to_html(self) -> str:
        parts = ["<table border='1'>"]
        for row in self.rows:
            parts.append("  <tr>")
            for cell in row:
                if cell.content is None:
                    continue
                attributes = ""
                if cell.row_span > 1:
                    attributes += f' rowspan="{cell.row_span}"'
                if cell.col_span > 1:
                    attributes += f' colspan="{cell.col_span}"'
                parts.append(f"    <td{attributes}>{html_lib.escape(cell.content)}</td>")
            parts.append("  </tr>")
        parts.append("</table>")
        return "\n".join(parts)


def build_html_table(df_word: pd.DataFrame, merge_close_words: bool = False) -> str:
    """
    Top-level entry point: turn a flat word-level OCR DataFrame (as produced
    by either PaddleTextRecognizer or AzureTextRecognizer) into an HTML table.
    """
    if df_word is None or df_word.empty:
        return "<table border='1'><tr><td>No text detected</td></tr></table>"

    rows = group_words_into_rows(df_word)
    if not rows:
        return "<table border='1'><tr><td>No text detected</td></tr></table>"

    cell_width = df_word['boundingBox'].map(lambda b: b[2] - b[0]).mean()
    cell_height = df_word['boundingBox'].map(lambda b: b[3] - b[1]).mean()
    cell_width = cell_width if cell_width and cell_width > 0 else 1
    cell_height = cell_height if cell_height and cell_height > 0 else 1

    _, num_cols, _, _ = compute_table_dimensions(df_word, cell_height, cell_width)
    num_cols = min(max(num_cols + 1, 1), MAX_GRID_COLUMNS)
    num_rows = len(rows)

    grid = Grid(num_rows, num_cols, cell_width, cell_height)
    grid.build(rows, merge_close_words=merge_close_words)
    return grid.to_html()
