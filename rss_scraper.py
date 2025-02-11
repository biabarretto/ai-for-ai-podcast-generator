import feedparser
import pandas as pd
from datetime import datetime, timedelta
from dateutil import parser

from utils import *
from models import ScrapedArticle


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
rss_urls = [ "https://news.mit.edu/topic/mitartificial-intelligence2-rss.xml",
"https://towardsdatascience.com/feed/",
"https://www.marktechpost.com/feed/",
"https://www.unite.ai/feed/", ]

for url in rss_urls:

    feed = feedparser.parse(url)
    source = feed.feed.title if "title" in feed.feed else "Unknown Source"
    print(f"Scraping articles from {source}")

    # Identify yesterday's date
    current_date = datetime.utcnow().date()
    threshold_date = current_date - timedelta(days=1)
    week = "10/02 - 16/02"

    # List to store extracted articles
    articles = []
    total_n_articles = 0

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

        # Stop iteration if the article is older than 7 days
        if published_date < threshold_date:
            print("Scraped all relevant articles")
            break  # RSS is chronological, so we can safely exit the loop

        # Extract article details from the RSS feed itself
        title = entry.title
        link = entry.link
        categories = [category.text for category in entry.categories] if "categories" in entry else []
        description = clean_html_content(entry.summary)
        # extract content and clean it using html parser
        content = entry.content[0].value if "content" in entry else "No content available"
        clean_content = clean_html_content(content)

        try:
            article = ScrapedArticle(
                source=source,
                link=link,
                title=title,
                category=categories,
                description=description,
                content=clean_content,
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


# Convert the list to a DataFrame
# df = pd.DataFrame(articles, columns=["Title", "Date_Published", "Link", "Description", "Content"])
# df['Date_Scraped'] = current_date


