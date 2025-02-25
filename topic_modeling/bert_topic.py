from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
import pandas as pd
import json


from utils import get_articles


class TopicModelingPipeline:
    """Pipeline for retrieving, processing, and analyzing articles using BERTopic."""

    def __init__(self, week_value):
        self.week_value = week_value
        self.articles = []
        self.texts = []
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.topic_model = BERTopic()
        self.df = None
        self.top_topics = None
        self.articles_by_topic = {}

    def load_articles(self):
        """Retrieve and prepare articles for topic modeling."""
        self.articles, self.texts = get_articles(self.week_value)
        print(f"Loaded {len(self.articles)} articles for week: {self.week_value}")

    def generate_embeddings(self):
        """Generate embeddings for the articles using SentenceTransformer."""
        print("Generating embeddings...")
        return self.model.encode(self.texts, show_progress_bar=True)

    def fit_model(self, embeddings):
        """Fit BERTopic model to the text embeddings."""
        print("Fitting BERTopic model...")
        topics, _ = self.topic_model.fit_transform(self.texts, embeddings)

        # Store results in a DataFrame
        self.df = pd.DataFrame({
            "Topic": topics,
            "Title": [article.title for article in self.articles],
            "Link": [article.link for article in self.articles],
            "Content": self.texts
        })

    def identify_top_topics(self, num_topics=3):
        """Identify the top topics based on frequency and print meaningful summaries."""
        self.top_topics = self.topic_model.get_topic_freq().head(num_topics)

        print("\n🔹 Top Identified Topics:\n")

        for topic in self.top_topics["Topic"]:
            # Get representative words for the topic
            topic_words = self.topic_model.get_topic(topic)
            topic_keywords = ", ".join([word[0] for word in topic_words[:5]])  # Top 5 keywords

            print(f"📌 **Topic {topic}:** {topic_keywords}")

            # Display a few representative articles (limit to 2 for readability)
            rep_docs = self.topic_model.get_representative_docs(topic)
            print("   🔹 Example Articles:")
            for i, doc in enumerate(rep_docs[:2]):  # Show only first 2 articles
                preview = " ".join(doc.split()[:50]) + "..."  # Show first 50 words
                print(f"     {i + 1}. {preview}\n")

        print("\n")

    def save_articles_by_topic(self):
        """Group articles by top topics and save them in JSON format for NotebookLM."""
        # Get all articles from top topics
        top_articles = self.df[self.df["Topic"].isin(self.top_topics["Topic"])]

        # Create dict with all articles per topic, including title and link
        self.articles_by_topic = {
            topic: top_articles[top_articles["Topic"] == topic][["Title", "Link", "Content"]].to_dict(orient="records")
            for topic in self.top_topics["Topic"]
        }

        # Save each topic as a json document with all articles
        for topic, articles in self.articles_by_topic.items():
            formatted_articles = [
                {
                    "title": article["Title"],
                    "link": article["Link"],
                    "content": article["Content"],
                    "source": f"Source: {article['Title']} ({article['Link']})"  # specifies source for each content
                }
                for article in articles
            ]

            filename = f"topic_{topic}.json"
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(formatted_articles, f, indent=4, ensure_ascii=False)

            print(f"Saved {filename} for NotebookLM summarization.")

    def visualize_topics(self):
        """Generate visualizations for topic modeling."""
        self.topic_model.visualize_barchart(top_n_topics=3)
        self.topic_model.visualize_topics()

    def run_pipeline(self):
        """Execute the entire topic modeling pipeline."""
        self.load_articles()
        embeddings = self.generate_embeddings()
        self.fit_model(embeddings)
        self.identify_top_topics()
        self.visualize_topics()
        self.save_articles_by_topic()


# Execute the pipeline
if __name__ == "__main__":
    week_value = "10/02 - 16/02"
    pipeline = TopicModelingPipeline(week_value)
    pipeline.run_pipeline()

