import re
import unicodedata


class TextCleaner:
    """
    Utility class for cleaning and preprocessing raw PDF text.
    """

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean and preprocess text by removing noise, fixing formatting, normalizing whitespace, and removing bullet points.
        Args:
            text (str): Raw text extracted from PDF.
        Returns:
            str: Cleaned text.
        """

        # Normalize Unicode characters (NFKC normalizes and composes characters)
        text = unicodedata.normalize("NFKC", text)

        # Remove non-printable/control characters (including extended ASCII control chars)
        text = re.sub(r"[\x00-\x1F\x7F-\x9F]", "", text)

        # Remove zero-width characters and other invisible Unicode
        text = re.sub(r"[\u200B-\u200D\uFEFF]", "", text)

        # Remove common bullet points and list markers at the start of lines
        text = re.sub(r"^[\s]*([\-\*•◦‣▪‣‣●○▪▫‣⁃])\s*", "", text, flags=re.MULTILINE)

        # Remove common PDF artifacts and formatting characters
        text = re.sub(r"[■□▪▫▬▭▮▯▰▱]", "", text)  # Box drawing characters
        text = re.sub(r"[←↑→↓↔↕↖↗↘↙]", "", text)  # Arrow characters
        text = re.sub(r"[♠♣♥♦]", "", text)  # Card suit characters

        # Remove common PDF text artifacts
        text = re.sub(
            r"[^\x20-\x7E\u00A0-\u00FF\u0100-\u017F\u0180-\u024F\u1E00-\u1EFF\u2C60-\u2C7F\uA720-\uA7FF]",
            "",
            text,
        )

        # Normalize whitespace (multiple spaces, tabs, newlines)
        text = re.sub(r"\s+", " ", text)

        # Remove leading/trailing whitespace
        text = text.strip()

        # Remove empty lines and excessive line breaks
        text = re.sub(r"\n\s*\n", "\n", text)

        # Final cleanup - remove any remaining control characters
        text = "".join(char for char in text if unicodedata.category(char)[0] != "C")

        return text

    @staticmethod
    def clean_text_aggressive(text: str) -> str:
        """
        More aggressive text cleaning for problematic PDFs.
        Args:
            text (str): Raw text extracted from PDF.
        Returns:
            str: Cleaned text.
        """

        # Apply basic cleaning first
        text = TextCleaner.clean_text(text)

        # Remove any remaining non-ASCII characters (more aggressive)
        text = re.sub(r"[^\x20-\x7E]", " ", text)

        # Remove excessive punctuation
        text = re.sub(r"[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}]", " ", text)

        # Normalize whitespace again
        text = re.sub(r"\s+", " ", text)
        text = text.strip()

        return text
