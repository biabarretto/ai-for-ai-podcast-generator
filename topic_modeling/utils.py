import re
import sqlite3
from datetime import datetime

from data_model.database import DB_PATH
from data_model.models import ScrapedArticle

def clean_text(text):
    "Cleans text data, removing markdown symbols, punctuation, and converting to lower-case"
    text = re.sub(r'\[.*?\]\(.*?\)', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    return text


def get_articles(week_value):
    """Retrieve articles from the database for a specific week and format them for topic modeling.
    Returns:
        articles: list with all ScrapedArticle pydantic objects for the given week
        texts: list with all content texts to be fed to model
        """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        "SELECT source, link, title, category, description, content, pub_date, scraped_date, week "
        "FROM articles WHERE week = ?", (week_value,)
    )
    rows = cursor.fetchall()
    conn.close()

    articles = []
    texts = []  # To store cleaned article text for topic modeling

    for row in rows:
        article = ScrapedArticle(
            source=row[0],
            link=row[1],
            title=row[2],
            category=row[3].split(", "),  # Convert back to list
            description=row[4],
            content=row[5],
            pub_date=datetime.fromisoformat(row[6]),
            scraped_date=datetime.fromisoformat(row[7]),
            week=row[8]
        )
        articles.append(article)

        # Merge title and content for topic modeling (not description as usually is already included in content)
        full_text = f"{article.title} {article.content}"
        texts.append(clean_text(full_text))  # Apply text cleaning

    return articles, texts


