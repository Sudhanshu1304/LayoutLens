"""
The contract you need to satisfy to plug your own OCR engine into this
project. `TextRecognizerProtocol` is a `typing.Protocol` — structural
typing, so you do NOT need to inherit from anything here. Any object with
a matching `.recognize(...)` method satisfies it; `PaddleTextRecognizer`
and `AzureTextRecognizer` already conform, with no changes.
"""
from pathlib import Path
from typing import List, Protocol, Union, runtime_checkable

import pandas as pd


@runtime_checkable
class TextRecognizerProtocol(Protocol):
    """
    A text recognizer performs word-level OCR on a whole document image
    and returns every word it found along with that word's bounding box.
    The Document Layout Parser (see `html_table_builder.build_html_table`)
    relies on this word-level granularity — a recognizer that only returns
    line- or paragraph-level text will not work here.
    """

    def recognize(self, image_path: Union[str, Path]) -> List[pd.DataFrame]:
        """
        Args:
            image_path: path to the input document image.

        Returns:
            A list of DataFrames, each with columns `text` (str) and
            `boundingBox` (`[xmin, ymin, xmax, ymax]` ints) — one row per
            word found, in the coordinate space of the original image.
            This project only ever uses the first DataFrame in the list.
        """
        ...
