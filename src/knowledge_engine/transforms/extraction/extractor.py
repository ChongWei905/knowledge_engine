import io
from abc import ABC, abstractmethod

import cv2
import fitz
import logging

import numpy as np
import pytesseract
from PIL import Image
from fitz import Page, Matrix

logger = logging.getLogger(__name__)

class Extractor(ABC):

    @abstractmethod
    def extract(self, pdf_bytes: bytes) -> str:
        pass


class EnhancedPDFTextExtractor(Extractor):
    """Extract text from PDF files using PyMuPDF and OCR, refers to a private project called ferry."""

    def __init__(self, context: bool = True, type: str = "pymupdf"):
        self.context = context
        self.type = type

    def extract(self, pdf_bytes: bytes) -> str:
        if self.type == "pymupdf":
            return self._extract_with_pymupdf(pdf_bytes)
        else:
            raise ValueError(f"Unsupported type: {self.type}")


    def _clean_text(self, text: str) -> str:
        if not text.strip():
            return ""

        # todo: fill the clean process
        return text

    def _extract_with_pymupdf(self, pdf_bytes: bytes):
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            text_parts = []

            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()

                if text.strip():
                    if self.context:
                        text_parts.append(f'[Page {page_num + 1}]')

                    text_parts.append(self._clean_text(text))
                else:
                    logger.warning(f"Page {page_num + 1} has no extractable text")
                    try:
                        ocr_text = self._ocr_page(page)
                        if ocr_text.strip():
                            if self.context:
                                text_parts.append(f'[Page {page_num + 1}]')
                            text_parts.append(ocr_text)
                    except Exception as e:
                        logger.warning(f"Fail when trying OCR: {str(e)}")

            doc.close()
            return '\n\n'.join(text_parts)
        except Exception as e:
            logger.error(f'PyMuPDF extraction failed: {str(e)}')
            return ""

    def _ocr_page(self, page: Page):
        try:
            mat = Matrix(2.0, 2.0)
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")

            image = Image.open(io.BytesIO(img_data))

            opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)

            denoised = cv2.fastNlMeansDenoising(gray)

            _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            custom_config = r'--oem 3 --psm 6 -l chi_sim+eng'
            text = pytesseract.image_to_string(binary, config=custom_config)

            return self._clean_text(text)

        except Exception as e:
            logger.error(f'OCR failed: {str(e)}')
            return ""
