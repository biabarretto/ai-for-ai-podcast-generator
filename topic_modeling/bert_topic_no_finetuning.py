from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
import pandas as pd
import json
from utils import get_articles, evaluate_coherence, evaluate_diversity_redundancy, evaluate_stability
from sklearn.feature_extraction.text import CountVectorizer
import os
from hdbscan import HDBSCAN
from umap import UMAP
import numpy as np
import random

random.seed(42)
np.random.seed(42)
class TopicModelingPipeline:
    """Pipeline for retrieving, processing, and analyzing articles using BERTopic."""

    def __init__(self, week_value):
        self.week_value = week_value
        self.articles = []
        self.texts = []
        # model = SentenceTransformer("all-mpnet-base-v2")  todo: test this embedding model which might be best for nuanced AI text
        self.embedding_model = 'all-MiniLM-L6-v2'
        self.model = SentenceTransformer(self.embedding_model)

        stop_words = ["ai", "language", "model", "models", "data", "tools", "research", "million",
                      "intelligence", "researchers", "interview", "series"]
        vectorizer = CountVectorizer(
            stop_words=stop_words,  # exclude generic words when extracting top-n words for each topic
            ngram_range=(1, 2),  # capture "language model", "deep learning"
            min_df=2,  # ignore words that appear only once
            max_df=0.8  # ignore overly common terms
        )
        # bert topic with hdbscan and umap
        hdbscan_model = HDBSCAN(min_cluster_size=5, min_samples=1, prediction_data=True)
        self.topic_model = BERTopic(
            embedding_model=self.model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer,
            umap_model=UMAP(n_neighbors=5, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
        )

        self.df = None
        self.top_topics = None
        self.articles_by_topic = {}

    def load_articles(self):
        """Retrieve and prepare articles for topic modeling."""
        self.articles, self.texts = get_articles(self.week_value)
        # Remove invalid or empty text entries
        valid_texts_with_indices = [(i, t) for i, t in enumerate(self.texts) if isinstance(t, str) and t.strip()]
        valid_indices, self.texts = zip(*valid_texts_with_indices) if valid_texts_with_indices else ([], [])

        # Filter corresponding articles to stay aligned
        self.articles = [self.articles[i] for i in valid_indices]
        print(f"Loaded {len(self.articles)} articles for week: {self.week_value}")

    def generate_embeddings(self):
        """Generate or load embeddings for one or multiple weeks of articles."""
        week_values = self.week_value if isinstance(self.week_value, list) else [self.week_value]
        all_embeddings = []

        # Ensure the embeddings folder exists
        os.makedirs("embeddings", exist_ok=True)

        for week in week_values:
            safe_week = week.replace("/", "-")
            embedding_path = os.path.join("embeddings", f"embeddings_{self.embedding_model}_{safe_week}.npy")

            week_texts = [text for article, text in zip(self.articles, self.texts) if article.week == week]

            if os.path.exists(embedding_path):
                print(f"📂 Loading cached embeddings for {week}...")
                embeddings = np.load(embedding_path)
            else:
                print(f"⚙️ Generating embeddings for {week}...")
                embeddings = self.model.encode(week_texts, show_progress_bar=True)
                np.save(embedding_path, embeddings)

            all_embeddings.extend(embeddings)

        return np.array(all_embeddings)

    def fit_model(self, embeddings):
        """Fit BERTopic model to the text embeddings."""
        print("Fitting BERTopic model...")
        topics, _ = self.topic_model.fit_transform(self.texts, embeddings)
        # self.topic_model.reduce_frequent_words(threshold=0.5)  # removes words that appear in more than 50% of topics
        # would need to update bert in order to use this

        assert len(self.texts) == len(self.articles) == len(topics), "Mismatch in lengths after filtering!"
        # Store results in a DataFrame
        self.df = pd.DataFrame({
            "Topic": topics,
            "Title": [article.title for article in self.articles],
            "Link": [article.link for article in self.articles],
            "Content": self.texts
        })

    def identify_top_topics(self, num_topics=3):
        """Identify the top topics based on frequency and print meaningful summaries."""
        topics = self.topic_model.get_topic_freq()
        print(f"{len(topics)} topics identified")
        self.top_topics = topics.head(num_topics)
        print("\n🔹 Top Topics:\n")
        for topic in self.top_topics["Topic"]:
            topic_words = self.topic_model.get_topic(topic)
            topic_keywords = ", ".join([word[0] for word in topic_words[:8]])
            print(f"📌 **Topic {topic}:** {topic_keywords}")

            rep_docs = self.topic_model.get_representative_docs(topic)
            print("   🔹 Example Articles:")
            for i, doc in enumerate(rep_docs[:2]):
                preview = " ".join(doc.split()[:50]) + "..."
                print(f"     {i + 1}. {preview}\n")
        print("\n")

    def evaluate_model(self):
        evaluate_coherence(self.topic_model, self.texts)
        evaluate_diversity_redundancy(self.topic_model)
        evaluate_stability(self.topic_model, self.texts, self.model)

    def save_articles_by_topic(self, save=True):
        """Group articles by top topics and save them in JSON format for NotebookLM."""
        top_articles = self.df[self.df["Topic"].isin(self.top_topics["Topic"])]
        self.articles_by_topic = {
            topic: top_articles[top_articles["Topic"] == topic][["Title", "Link", "Content"]].to_dict(orient="records")
            for topic in self.top_topics["Topic"]
        }

        for topic, articles in self.articles_by_topic.items():
            formatted_articles = [
                {
                    "title": article["Title"],
                    "link": article["Link"],
                    "content": article["Content"],
                    "source": f"Source: {article['Title']} ({article['Link']})"
                }
                for article in articles
            ]
            filename = f"topic_{topic}.json"
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(formatted_articles, f, indent=4, ensure_ascii=False)
            print(f"Saved {filename} for NotebookLM summarization.")

    def visualize_topics(self, run_number):
        """Generate visualizations and save them to charts/ subfolder."""
        os.makedirs("charts", exist_ok=True)

        bar_chart = self.topic_model.visualize_barchart()
        bar_chart.write_html(f"charts/bar_chart_run{run_number}.html")

        topics_vis = self.topic_model.visualize_topics()
        topics_vis.write_html(f"charts/topic_map_run{run_number}.html")

    def run_pipeline(self, save=True):
        """Execute the entire topic modeling pipeline."""
        self.load_articles()
        embeddings = self.generate_embeddings()
        self.fit_model(embeddings)
        self.identify_top_topics(num_topics=10)
        self.evaluate_model()
        self.visualize_topics(run_number=7)
        if save:
            self.save_articles_by_topic()

# Execute the pipeline
if __name__ == "__main__":
    # week_value = ["10/02 - 16/02", "17/02 - 23/02", "24/02 - 02/03"]
    week_value = "17/02 - 23/02"
    pipeline = TopicModelingPipeline(week_value)
    pipeline.run_pipeline(save=False)
