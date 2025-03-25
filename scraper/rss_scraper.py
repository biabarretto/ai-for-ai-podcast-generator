import feedparser
from datetime import datetime, timedelta
from dateutil import parser
from markdownify import markdownify as md
import re

from scraper.utils import *
from data_model.models import ScrapedArticle


rss_urls = [ "https://towardsdatascience.com/feed/",
    "https://news.mit.edu/topic/mitartificial-intelligence2-rss.xml",
    "https://www.marktechpost.com/feed/",
    "https://www.unite.ai/feed/", ]

# Identify yesterday's date
current_date = datetime.utcnow().date()
threshold_date = current_date - timedelta(days=1)

# Define week value
week_start, week_end = get_week_range(current_date)
week = f"{week_start} - {week_end}"
print(f"Scraping articles from {threshold_date} for week {week}")

total_n_articles = 0

for url in rss_urls:

    feed = feedparser.parse(url)
    source = feed.feed.title if "title" in feed.feed else "Unknown Source"
    print(f"Scraping articles from {source}")

    # List to store extracted articles
    articles = []

    for entry in feed.entries:
        try:
            # Try parsing using the expected format with timezone
            published_date = datetime.strptime(entry.published, "%a, %d %b %Y %H:%M:%S %z")
        except ValueError:
            try:
                # Fallback to dateutil.parser for unknown formats
                published_date = parser.parse(entry.published)
            except Exception as e:
                print(f"Error parsing date for {entry.title}: {e}")
                break  # Stop if date parsing fails

        # Convert to naive UTC (remove timezone info)
        published_date = published_date.replace(tzinfo=None).date()

        # Scrape while pub_date is after the threshold_date
        if published_date >= threshold_date:
            # Extract article details from the RSS feed itself
            title = entry.title
            link = entry.link
            categories = [tag["term"] for tag in getattr(entry, "tags", []) if "term" in tag]
            description = remove_post_footer(entry.summary)
            description = clean_html_content(description)
            content = entry.content[0].value if "content" in entry else "No content available"
            # Convert HTML content to Markdown
            markdown_content = md(content)
            # Remove Markdown images (e.g., ![alt text](image.png))
            markdown_content_clean = re.sub(
                r'!\[.*?\]\((https?:\/\/[^\)]+\.(?:png|jpg|jpeg|gif|svg|webp|bmp|ico|tiff))\)', '',
                markdown_content, flags=re.IGNORECASE)
            # Remove standard Markdown links but keep their text (excluding images)
            markdown_content_clean = re.sub(r'\[([^\]]+)\]\((https?:\/\/[^\)]+)\)', r'\1',
                                            markdown_content_clean)

            try:
                article = ScrapedArticle(
                    source=source,
                    link=link,
                    title=title,
                    category=categories,
                    description=description,
                    content=markdown_content_clean,
                    pub_date=published_date,
                    scraped_date=current_date,
                    week=week
                )
                articles.append(article)
                print(f"Processed article {title}")
            except Exception as e:
                print(f"Error processing article {link} - {e}")



    # Save articles in db
    insert_articles(articles)
    print(f"Inserted {len(articles)} articles")
    total_n_articles += len(articles)

print(f"Scraped {total_n_articles} articles")



