from bs4 import BeautifulSoup
import re

__all__ = ["clean_html_content"]  # Specify what to export when calling file

def clean_html_content(html_content):
    """
    Cleans the provided HTML content by removing tags, symbols, and unnecessary spaces.
    """
    # Use BeautifulSoup to parse the HTML and extract text
    soup = BeautifulSoup(html_content, "html.parser")
    clean_text = soup.get_text(separator=" ")  # Extract text and replace tags with spaces

    # Remove extra spaces, HTML entities, and non-alphanumeric symbols (if needed)
    clean_text = re.sub(r'\s+', ' ', clean_text)  # Replace multiple spaces with a single space
    clean_text = re.sub(r'&[a-z]+;', '', clean_text)  # Remove HTML entities like &nbsp;
    clean_text = re.sub(r'[^\w\s.,]', '', clean_text)  # Keep periods and commas

    return clean_text.strip()
