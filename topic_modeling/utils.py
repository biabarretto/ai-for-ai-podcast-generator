import re

def clean_text(text):
    "Cleans text data, removing markdown symbols, punctuation, and converting to lower-case"
    text = re.sub(r'\[.*?\]\(.*?\)', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    return text

