import feedparser
import pandas as pd
from datetime import datetime, timedelta

from utils import *


# Define the RSS feed URL
rss_url = "https://news.mit.edu/topic/mitartificial-intelligence2-rss.xml"

# Parse the RSS feed
feed = feedparser.parse(rss_url)

# Get the current date and compute the threshold date for filtering (last 7 days)
current_date = datetime.utcnow()
threshold_date = current_date - timedelta(days=7)

# List to store extracted articles
articles = []

# Iterate over each entry in the RSS feed
for entry in feed.entries:
    # Convert published date from RSS format: "Tue, 04 Feb 2025 00:00:00 -0500"
    try:
        published_date = datetime.strptime(entry.published, "%a, %d %b %Y %H:%M:%S %z")
        published_date = published_date.replace(tzinfo=None)  # Convert to naive UTC
    except Exception as e:
        print(f"Error parsing date for {entry.title}: {e}")
        break  # Stop if date parsing fails

    # Stop iteration if the article is older than 7 days
    if published_date < threshold_date:
        break  # RSS is chronological, so we can safely exit the loop

    # Extract article details from the RSS feed itself
    title = entry.title
    link = entry.link
    description = entry.summary
    # extract content and clean it using html parser
    content = entry.content[0].value if "content" in entry else "No content available"
    clean_content = clean_html_content(content)


    # Append data to the list
    articles.append([title, published_date.strftime("%Y-%m-%d"), link, description, content])

# Convert the list to a DataFrame
df = pd.DataFrame(articles, columns=["Title", "Date_Published", "Link", "Description", "Content"])
df['Date_Scraped'] = current_date

# Display the table
import ace_tools as tools
tools.display_dataframe_to_user(name="MIT AI News Articles", dataframe=df)

# Save to CSV (optional)
# df.to_csv("mit_ai_news.csv", index=False)

