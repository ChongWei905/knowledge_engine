import logging
from typing import Any


logger = logging.getLogger(__name__)

def get_prompt(
    processor: Any,
    raw_prompt: str,
    has_image: bool = False,
    image_min_pixels: int = 224 * 224,
    image_max_pixels: int = 1280 * 28 * 28
):
    content = [
        {
            "type": "text",
            "text": raw_prompt
        }
    ]
    if has_image:
        content.append({
            "type": "image",
            "min_pixels": image_min_pixels,
            "max_pixels": image_max_pixels
        })
    message = {"role": "user", "content": content}
    try:
        prompt = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        return prompt
    except ValueError as e:
        logger.warning(f"Got error {e} trying to apply chat template. Using raw prompt instead.")
        return raw_prompt
