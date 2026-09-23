from pathlib import Path
from typing import List, Optional, Union
import numpy as np
import pandas as pd
from paddleocr import PaddleOCR
from PIL import Image


class PaddleTextRecognizer:
    """
    Word-level OCR on a whole document image, using PaddleOCR. Runs fully
    locally — no account or API key needed. Satisfies
    `recognizers.protocols.TextRecognizerProtocol`.

    Attributes:
        models_dir (Path): Directory containing the PaddleOCR model files
            (defaults to the bundled `paddleocr_models/` next to this file).
    """

    def __init__(self, models_dir: Optional[Union[str, Path]] = None) -> None:
        self.models_dir = Path(models_dir) if models_dir else Path(__file__).parent / 'paddleocr_models'
        self._setup_model_dirs()

        self.model = PaddleOCR(
            use_angle_cls=False,
            lang='en',
            det_model_dir=str(self.models_dir / 'det'),
            rec_model_dir=str(self.models_dir / 'rec'),
        )

    def _setup_model_dirs(self) -> None:
        (self.models_dir / 'det').mkdir(parents=True, exist_ok=True)
        (self.models_dir / 'rec').mkdir(parents=True, exist_ok=True)

    def recognize(self, image_path: Union[str, Path]) -> List[pd.DataFrame]:
        """Run OCR on the whole image and return one word-level DataFrame."""
        with Image.open(image_path) as img:
            img_array = np.array(img.convert('RGB'))

        ocr_result = self.model.ocr(img_array)

        processed_data = [
            (item[1][0], [
                np.array(item[0])[:, 0].min(),
                np.array(item[0])[:, 1].min(),
                np.array(item[0])[:, 0].max(),
                np.array(item[0])[:, 1].max(),
            ])
            for item in ocr_result[0]
        ]

        return [pd.DataFrame(
            sorted(processed_data, key=lambda x: (x[1][1], x[1][0])),
            columns=['text', 'boundingBox'],
        )]
