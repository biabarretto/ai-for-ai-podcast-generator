from bs4 import BeautifulSoup
import re
import sqlite3

from database import DB_NAME
from models import ScrapedArticle

__all__ = ["clean_html_content", "insert_articles", "get_articles"]  # Specify what to export when calling file

def insert_articles(articles: list[ScrapedArticle]):
    """Insert a list of articles into SQLite."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    try:
        # Prepare the data for insertion
        article_data = [
            (
                article.source,
                article.link,
                article.title,
                article.category,  # Already converted to a string
                article.description,
                article.content,
                article.pub_date.isoformat(),
                article.scraped_date.isoformat()
            )
            for article in articles
        ]

        # Use executemany to insert multiple records at once
        cursor.executemany("""
        INSERT INTO articles (source, link, title, category, description, content, pub_date, scraped_date)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, article_data)

        conn.commit()
    except sqlite3.IntegrityError:
        print("Skipping duplicate entry due to IntegrityError.")
    finally:
        conn.close()



def get_articles():
    """Retrieve all stored articles from the database."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("SELECT source, link, title, category, description, content, pub_date, scraped_date FROM articles")
    rows = cursor.fetchall()

    articles = []
    for row in rows:
        article = ScrapedArticle(
            source=row[0],
            link=row[1],
            title=row[2],
            category=row[3].split(", "),  # Convert back to list
            description=row[4],
            content=row[5],
            pub_date=datetime.fromisoformat(row[6]),
            scraped_date=datetime.fromisoformat(row[7])
        )
        articles.append(article)

    conn.close()
    return articles


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
