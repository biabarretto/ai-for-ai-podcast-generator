import feedparser
from datetime import datetime, timedelta
from dateutil import parser
from markdownify import markdownify as md
import re

from scraper.utils import *
from data_model.models import ScrapedArticle


# Define the RSS feed URL
# rss_urls = [ "https://news.mit.edu/topic/mitartificial-intelligence2-rss.xml",
# "https://aimagazine.com/api/multi-feed?feedType=rss&limit=10&contentType=report&paged=1&paged=1",
# "https://news.berkeley.edu/category/research/technology-engineering/feed/",
# "https://techcrunch.com/feed/",
# "https://www.wired.com/feed/tag/ai/latest/rss",
# "https://techxplore.com/rss-feed/machine-learning-ai-news/",
# "https://towardsdatascience.com/feed/",
# "https://www.marktechpost.com/feed/",
# "https://www.unite.ai/feed/", ]
rss_urls = [ "https://towardsdatascience.com/feed/",
    "https://news.mit.edu/topic/mitartificial-intelligence2-rss.xml",
    "https://www.marktechpost.com/feed/",
    "https://www.unite.ai/feed/", ]

# Identify yesterday's date
current_date = datetime.utcnow().date()
threshold_date = current_date - timedelta(days=1)
print(f"Scraping articles from {threshold_date}")
# Define week value
week = "10/02 - 16/02"

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
            description = clean_html_content(entry.summary)
            content = entry.content[0].value if "content" in entry else "No content available"
            # Convert HTML content to Markdown
            markdown_content = md(content)
            markdown_content_clean = re.sub(r'\[([^\]]+)\]\((https?:\/\/[^\)]+)\)', r'\1', markdown_content)

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



