import sys
import os
from typing import Optional
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def detect_language(text: str) -> str:
    """Detect language of input text."""
    try:
        from googletrans import Translator
        translator = Translator()
        detected = translator.detect(text)
        return detected.lang
    except Exception:
        return "en"  # Default to English


def translate_text(text: str, target_lang: str = "en") -> str:
    """Translate text to target language."""
    try:
        from googletrans import Translator
        translator = Translator()
        translated = translator.translate(text, dest=target_lang)
        return translated.text
    except Exception:
        return text  # Return original if translation fails


def translate_query(query: str) -> tuple[str, str]:
    """Translate query to English if needed, return (translated_query, detected_lang)."""
    try:
        detected = detect_language(query)
        if detected != "en":
            translated = translate_text(query, "en")
            return translated, detected
        return query, "en"
    except Exception:
        return query, "en"


def translate_response(response: str, target_lang: str) -> str:
    """Translate response to target language."""
    if target_lang == "en":
        return response
    try:
        return translate_text(response, target_lang)
    except Exception:
        return response

