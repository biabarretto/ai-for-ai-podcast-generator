import feedparser
from datetime import datetime, timedelta
from dateutil import parser
from markdownify import markdownify as md
import re

from scraper.utils import *
from data_model.models import ScrapedArticle

class RSSScraper:
    """
        Class to scrape articles from multiple RSS feeds and store them in a database.

        Attributes:
            urls (list): List of RSS feed URLs to scrape.
            current_date (date): The current UTC date.
            threshold_date (date): The date threshold to filter articles (articles from yesterday onwards).
            week (str): The week range in the format "YYYY-MM-DD - YYYY-MM-DD".
            total_articles (int): A count of the total number of articles scraped.
        """
    def __init__(self, urls):
        """
        Initializes the RSSScraper instance with the list of URLs and computes necessary date parameters.

        Args:
            urls (list): List of RSS feed URLs to scrape.
        """
        self.urls = urls
        self.current_date = datetime.utcnow().date()
        self.threshold_date = self.current_date - timedelta(days=1)
        self.week_start, self.week_end = get_week_range(self.current_date)
        self.week = f"{self.week_start} - {self.week_end}"
        self.total_articles = 0

    def scrape(self):
        """
        Scrapes articles from the provided RSS URLs and inserts them into the database.

        Iterates over each RSS feed, processes articles that are published from yesterday onwards,
        and saves them to the database.
        """
        print(f"Scraping articles from {self.threshold_date} for week {self.week}")
        for url in self.urls:
            feed = feedparser.parse(url)
            source = feed.feed.title if "title" in feed.feed else "Unknown Source"
            print(f"Scraping articles from {source}")

            # Parse each entry and process the data
            articles = self._parse_feed(feed, source)
            # Insert the articles into the database
            insert_articles(articles)

            self.total_articles += len(articles)
            print(f"Inserted {len(articles)} articles")

        print(f"Scraped {self.total_articles} articles")

    def _parse_feed(self, feed, source):
        """
        Parses the RSS feed entries and processes articles.

        Args:
            feed (FeedParserDict): The parsed RSS feed.
            source (str): The source name or title of the RSS feed.

        Returns:
            list: A list of ScrapedArticle objects.
        """

        articles = []
        for entry in feed.entries:
            try:
                # Parse the published date of the article
                published_date = self._parse_date(entry.published)
                # Skip articles that are older than the threshold date
                if published_date < self.threshold_date:
                    continue

                # Process the article and extract necessary details
                article = ScrapedArticle(
                    source=source,
                    link=entry.link,
                    title=entry.title,
                    category=[tag["term"] for tag in getattr(entry, "tags", []) if "term" in tag],
                    description=clean_html_content(remove_post_footer(entry.summary)),
                    content=self._clean_markdown(entry),
                    pub_date=published_date,
                    scraped_date=self.current_date,
                    week=self.week
                )
                articles.append(article)
                print(f"Processed article {entry.title}")
            except Exception as e:
                print(f"Error processing article {entry.link}: {e}")
        return articles

    def _parse_date(self, date_str):
        """ Tries to parse a string date and return a naive datetime object (no timezone) """
        try:
            # Try to parse using a common format with timezone information
            date = datetime.strptime(date_str, "%a, %d %b %Y %H:%M:%S %z")
        except ValueError:
            # Fallback to a more flexible parser from dateutil if the format is unknown
            date = parser.parse(date_str)

        return date.replace(tzinfo=None).date()

    def _clean_markdown(self, entry):
        """ Converts the article content from HTML to markdown and removes unwanted content such as images and links """
        content = entry.content[0].value if "content" in entry else "No content available"
        markdown_content = md(content)
        markdown_no_images = re.sub(r'!\[.*?\]\((https?:\/\/[^\)]+\.(?:png|jpg|jpeg|gif|svg|webp|bmp|ico|tiff))\)', '', markdown_content, flags=re.IGNORECASE)
        markdown_no_links = re.sub(r'\[([^\]]+)\]\((https?:\/\/[^\)]+)\)', r'\1', markdown_no_images)
        return markdown_no_links

if __name__ == "__main__":
    rss_urls = [
        "https://towardsdatascience.com/feed/",
        "https://news.mit.edu/topic/mitartificial-intelligence2-rss.xml",
        "https://www.marktechpost.com/feed/",
        "https://www.unite.ai/feed/",
    ]
    scraper = RSSScraper(rss_urls)
    scraper.scrape()


