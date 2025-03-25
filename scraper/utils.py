from bs4 import BeautifulSoup
import re
import sqlite3
from datetime import datetime, timedelta


from data_model.database import DB_PATH
from data_model.models import ScrapedArticle

__all__ = ["clean_html_content", "insert_articles", "get_articles", "get_week_range", "remove_post_footer"]  # Specify what to export when calling file

def insert_articles(articles: list[ScrapedArticle]):
    """Insert a list of articles into SQLite."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        # Prepare the data for insertion
        article_data = [
            (
                article.source,
                str(article.link),
                article.title,
                article.category_as_string(),  # Use method to convert to string
                article.description,
                article.content,
                article.pub_date.isoformat(),
                article.scraped_date.isoformat(),
                article.week
            )
            for article in articles
        ]

        # Use `INSERT OR IGNORE` to skip duplicates
        cursor.executemany("""
            INSERT OR IGNORE INTO articles 
            (source, link, title, category, description, content, pub_date, scraped_date, week)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, article_data)
        conn.commit()

    except Exception as e:
        print(f"Error inserting articles: {e}")
    finally:
        conn.close()



def get_articles():
    """Retrieve all stored articles from the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT source, link, title, category, description, content, pub_date, scraped_date, week FROM articles")
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
            scraped_date=datetime.fromisoformat(row[7]),
            week=row[8]
        )
        articles.append(article)

    conn.close()
    return articles


def remove_post_footer(html_content):
    soup = BeautifulSoup(html_content, "html.parser")

    # Find the <p> tag that starts with "The post"
    for p in soup.find_all("p"):
        if p.get_text(strip=True).startswith("The post"):
            p.decompose()  # remove the tag from the soup
            break

    return str(soup)

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


def get_week_range(date):
    """Given a date, return the start and end dates of the corresponding week (Monday-Sunday format)."""
    start_of_week = date - timedelta(days=date.weekday())  # Get Monday of the week
    end_of_week = start_of_week + timedelta(days=6)  # Get Sunday of the same week
    return start_of_week.strftime("%d/%m"), end_of_week.strftime("%d/%m")
