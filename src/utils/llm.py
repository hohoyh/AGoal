# src/utils/llm.py
import os
import time
import json
import base64
import logging
from io import BytesIO

from openai import OpenAI
import PIL

# avoid huggingface tokenizers fork warning spamming logs when multiprocessing is used
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# simple logger (you can configure logging.basicConfig in your main script)
logger = logging.getLogger("AGoal.LLM")
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(message)s"))
    logger.addHandler(ch)
logger.setLevel(logging.INFO)


def _extract_text_from_completion(resp):
    """
    Try multiple ways to extract textual content from an OpenAI-compatible completion/response.
    Returns str or None.
    """
    try:
        # attribute-style (openai-python object)
        if hasattr(resp, "choices"):
            choices = resp.choices
            if len(choices) == 0:
                return None
            choice0 = choices[0]
            # new-style: choice0.message.content
            if hasattr(choice0, "message"):
                msg = choice0.message
                # attribute
                if hasattr(msg, "content"):
                    return msg.content
                # dict-like
                if isinstance(msg, dict) and "content" in msg:
                    return msg["content"]
            # older-style: choice0.text
            if hasattr(choice0, "text"):
                return choice0.text

        # dict-like fallback (if the client returns plain dict)
        if isinstance(resp, dict):
            if "choices" in resp and len(resp["choices"]) > 0:
                c0 = resp["choices"][0]
                if isinstance(c0, dict):
                    # nested 'message' -> 'content'
                    if "message" in c0 and isinstance(c0["message"], dict) and "content" in c0["message"]:
                        # sometimes content is a list, join if so
                        content = c0["message"]["content"]
                        if isinstance(content, list):
                            try:
                                # some gateways use list of dicts: [{"type":"output_text","text":"..."}]
                                text_items = []
                                for it in content:
                                    if isinstance(it, dict):
                                        if "text" in it:
                                            text_items.append(it["text"])
                                        elif "content" in it:
                                            text_items.append(it["content"])
                                    elif isinstance(it, str):
                                        text_items.append(it)
                                return "\n".join(text_items) if text_items else str(content)
                            except Exception:
                                return str(content)
                        return str(content)
                    # fallback to 'text' field
                    if "text" in c0:
                        return str(c0["text"])
        # last resort: try to stringify object
        try:
            return str(resp)
        except Exception:
            return None
    except Exception as e:
        logger.debug(f"_extract_text_from_completion exception: {e}")
        try:
            return str(resp)
        except Exception:
            return None


class LLM:
    def __init__(self, base_url, api_key, llm_model, timeout=15, max_retries=3, backoff=1.0):
        self.base_url = base_url
        self.api_key = api_key
        self.llm_model = llm_model
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff = backoff

    def __call__(self, prompt):
        """
        Call chat completion. Returns a string (could be empty) but never None.
        On failure returns empty string "" and logs details.
        """
        client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.debug(f"LLM request attempt {attempt} model={self.llm_model}")
                chat_completion = client.chat.completions.create(
                    model=self.llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    # if your OpenAI client accepts request timeout, add it here as kwargs (depends on client version)
                )
                text = _extract_text_from_completion(chat_completion)
                if text is None:
                    logger.warning(f"[LLM] Empty/None response on attempt {attempt}. raw={repr(chat_completion)[:400]}")
                    # continue to retry
                else:
                    return text
            except Exception as e:
                # log exception with limited length of prompt (avoid logging secrets)
                logger.warning(f"[LLM] Exception on attempt {attempt}: {e}")
                # optionally log a truncated prompt for debug
                logger.debug(f"[LLM] prompt (truncated): {prompt[:500]}")
            # backoff
            if attempt < self.max_retries:
                time.sleep(self.backoff * attempt)
        # final fallback
        logger.error("[LLM] All attempts failed — returning empty string.")
        return ""


class VLM:
    def __init__(self, base_url, api_key, vlm_model, timeout=30, max_retries=2, backoff=1.0):
        self.base_url = base_url
        self.api_key = api_key
        self.vlm_model = vlm_model
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff = backoff

    def _image_to_base64(self, image):
        """
        Accept PIL.Image.Image or numpy array (H,W,3) or a path string.
        Return base64-encoded PNG string (no header).
        """
        if isinstance(image, str):
            img = PIL.Image.open(image).convert("RGB")
        elif hasattr(image, "save"):
            img = image.convert("RGB")
        else:
            # try to handle numpy arrays
            try:
                import numpy as np
                img = PIL.Image.fromarray(image.astype("uint8")).convert("RGB")
            except Exception as e:
                raise ValueError("Unsupported image type for VLM._image_to_base64") from e

        buffered = BytesIO()
        img.save(buffered, format="PNG")
        image_bytes = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return image_bytes

    def __call__(self, prompt, image):
        """
        Try sending a message that includes the prompt and an embedded data URL for the image.
        Returns string (possibly empty) but never None.
        """
        # prepare payload
        try:
            image_b64 = self._image_to_base64(image)
        except Exception as e:
            logger.error(f"[VLM] Failed to convert image: {e}")
            return ""

        # note: many custom gateways accept slightly different formats.
        # We keep message content as a list of items (text + image_url) like earlier code,
        # but we also try a plain textual fallback prompt if the gateway rejects structured content.
        client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.debug(f"VLM request attempt {attempt} model={self.vlm_model}")
                chat_completion = client.chat.completions.create(
                    model=self.vlm_model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + image_b64}},
                            ],
                        }
                    ],
                )
                text = _extract_text_from_completion(chat_completion)
                if text:
                    return text
                else:
                    logger.warning(f"[VLM] Empty response on attempt {attempt}.")
            except Exception as e:
                logger.warning(f"[VLM] Exception on attempt {attempt}: {e}")
                logger.debug(f"[VLM] prompt(trunc): {prompt[:300]}")
            if attempt < self.max_retries:
                time.sleep(self.backoff * attempt)

        # fallback: try plain-text prompt without structured image, giving the data-url inline after text
        try:
            fallback_prompt = prompt + "\n\nImage (data URL): data:image/png;base64," + image_b64[:300] + "...(truncated)"
            chat_completion = client.chat.completions.create(
                model=self.vlm_model,
                messages=[{"role": "user", "content": fallback_prompt}],
            )
            return _extract_text_from_completion(chat_completion) or ""
        except Exception as e:
            logger.error(f"[VLM] All attempts failed. Last exception: {e}")
            return ""
