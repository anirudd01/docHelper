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
        import re
        # Remove non-printable/control characters
        text = re.sub(r'[\x00-\x1F\x7F]', '', text)
        # Remove common bullet points at the start of lines
        text = re.sub(r'^[\s]*([\-\*•◦‣▪‣‣●○])\s+', '', text, flags=re.MULTILINE)
        # Normalize whitespace (multiple spaces, tabs, newlines)
        text = re.sub(r'\s+', ' ', text)
        # Strip leading/trailing whitespace
        text = text.strip()
        return text 