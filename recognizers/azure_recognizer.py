import io
import os
import time
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from PIL import Image
from msrest.authentication import CognitiveServicesCredentials
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes


class AzureCredentialsError(RuntimeError):
    """Raised when the Azure Computer Vision endpoint/key are missing or invalid."""


class AzureTextRecognizer:
    """
    Word-level OCR on a whole document image, backed by Azure AI Vision's
    Computer Vision `Read` API (v3.2 —
    https://{endpoint}/vision/v3.2/read/analyze). Satisfies
    `recognizers.protocols.TextRecognizerProtocol`.

    Internally this submits the image to Azure Read, polls the async
    operation until it finishes, then flattens Azure's own line-grouped
    response (lines, each containing words) into the flat one-row-per-word
    DataFrame this project expects.
    """

    # Free tier (F0): max 4 MB per request.
    MAX_IMAGE_BYTES = 4 * 1024 * 1024
    POLL_INTERVAL_SECONDS = 0.5
    POLL_TIMEOUT_SECONDS = 60

    def __init__(
        self,
        endpoint: Optional[str] = None,
        subscription_key: Optional[str] = None,
    ) -> None:
        """
        Args:
            endpoint: Azure Computer Vision resource endpoint, e.g.
                'https://<resource>.cognitiveservices.azure.com/'. Falls back
                to the AZURE_CV_ENDPOINT env var.
            subscription_key: Azure Computer Vision subscription key. Falls
                back to the AZURE_CV_KEY env var.
        """
        endpoint = endpoint or os.environ.get('AZURE_CV_ENDPOINT')
        subscription_key = subscription_key or os.environ.get('AZURE_CV_KEY')

        if not endpoint or not subscription_key:
            raise AzureCredentialsError(
                "Azure Computer Vision credentials are missing. Set AZURE_CV_ENDPOINT "
                "and AZURE_CV_KEY (e.g. in a .env file — see .env.example), or pass "
                "endpoint/subscription_key explicitly."
            )

        self.endpoint = endpoint
        self.client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))
        self.last_call_stats: dict = {}

    def recognize(self, image_path: Union[str, Path]) -> List[pd.DataFrame]:
        """Run OCR on the whole image via Azure Read and return one word-level DataFrame."""
        with Image.open(image_path) as img:
            img_array = np.array(img.convert('RGB'))

        words = self._raw_ocr(img_array)

        return [pd.DataFrame(
            sorted(words, key=lambda x: (x[1][1], x[1][0])),
            columns=['text', 'boundingBox'],
        )]

    def _raw_ocr(self, img_array: np.ndarray) -> List[tuple]:
        """
        Submit an image (as a numpy array) to Azure Read and return a flat
        list of (text, bbox) tuples, bbox = [xmin, ymin, xmax, ymax] ints.
        """
        image_bytes = self._encode_for_upload(img_array)

        start = time.perf_counter()
        read_response = self.client.read_in_stream(io.BytesIO(image_bytes), raw=True)
        read_operation_location = read_response.headers["Operation-Location"]
        operation_id = read_operation_location.split("/")[-1]

        waited = 0.0
        while True:
            read_result = self.client.get_read_result(operation_id)
            if read_result.status not in ('notStarted', 'running'):
                break
            time.sleep(self.POLL_INTERVAL_SECONDS)
            waited += self.POLL_INTERVAL_SECONDS
            if waited > self.POLL_TIMEOUT_SECONDS:
                raise TimeoutError(
                    f"Azure Read did not finish within {self.POLL_TIMEOUT_SECONDS}s "
                    f"(operation_id={operation_id})"
                )

        self.last_call_stats = {
            'elapsed_seconds': round(time.perf_counter() - start, 2),
            'status': read_result.status,
        }

        words: List[tuple] = []
        if read_result.status != OperationStatusCodes.succeeded:
            return words

        for text_result in read_result.analyze_result.read_results:
            self.last_call_stats['rotation_angle'] = text_result.angle
            for line in text_result.lines:
                for word in line.words:
                    bbox = self._polygon_to_xyxy(word.bounding_box)
                    words.append((word.text, bbox))

        return words

    def _encode_for_upload(self, img_array: np.ndarray) -> bytes:
        """Encode as PNG, falling back to a smaller JPEG if over the 4 MB free-tier limit."""
        buf = io.BytesIO()
        Image.fromarray(img_array).save(buf, format='PNG')
        data = buf.getvalue()
        if len(data) <= self.MAX_IMAGE_BYTES:
            return data

        for quality in (90, 75, 60, 45):
            buf = io.BytesIO()
            Image.fromarray(img_array).convert('RGB').save(buf, format='JPEG', quality=quality)
            data = buf.getvalue()
            if len(data) <= self.MAX_IMAGE_BYTES:
                return data

        raise ValueError(
            f"Image is {len(data) / 1024 / 1024:.1f} MB even at JPEG quality 45, "
            f"over Azure's {self.MAX_IMAGE_BYTES / 1024 / 1024:.0f} MB free-tier limit."
        )

    @staticmethod
    def _polygon_to_xyxy(bounding_box: List[float]) -> List[int]:
        """Azure returns an 8-value polygon [x1,y1,x2,y2,x3,y3,x4,y4]; convert to [xmin,ymin,xmax,ymax]."""
        xs = bounding_box[0::2]
        ys = bounding_box[1::2]
        return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]
