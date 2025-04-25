from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from bert_score import score
import pandas as pd
import json
from utils import get_articles, evaluate_coherence, evaluate_diversity_redundancy, evaluate_topic_quality_with_bertscore
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MinMaxScaler
from bertopic.vectorizers import ClassTfidfTransformer
import os
from hdbscan import HDBSCAN
from umap import UMAP
import numpy as np
import random

random.seed(42)
np.random.seed(42)
class TopicModelingPipeline:
    """Pipeline for retrieving, processing, and group articles using BERTopic."""

    def __init__(self, week_value, embedding_model='all-mpnet-base-v2'):
        self.week_value = week_value
        self.articles = []
        self.texts = []
        self.embedding_model = embedding_model
        self.model = SentenceTransformer(self.embedding_model)

        # define vectorizer
        stop_words = ["ai", "language", "model", "models", "data", "tools", "research", "million",
                      "intelligence", "researchers", "interview", "series"]
        vectorizer = CountVectorizer(
            stop_words=stop_words,  # exclude generic words when extracting top-n words for each topic
            ngram_range=(1, 2),  # capture "language model", "deep learning"
            min_df=0.03,  # exclude words that don't appear in at least 3% of documents
            max_df=0.8  # remove any word that appears in more than 80% of all documents
        )

        # bert topic with hdbscan and umap
        hdbscan_model = HDBSCAN(min_cluster_size=5, min_samples=1, prediction_data=True)
        self.topic_model = BERTopic(
            embedding_model=self.model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer,
            umap_model=UMAP(n_neighbors=5, n_components=5, min_dist=0.0, metric='cosine', random_state=42),
            ctfidf_model=ClassTfidfTransformer(reduce_frequent_words=True)
        )

        self.df = None
        self.top_topics = None
        self.articles_by_topic = {}

    def load_articles(self):
        """Retrieve and prepare articles for topic modeling."""
        # Retrieve articles from the db for the given week
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

        assert len(self.texts) == len(self.articles) == len(topics), "Mismatch in lengths after filtering!"
        # Store results in a DataFrame
        self.df = pd.DataFrame({
            "Topic": topics,
            "Title": [article.title for article in self.articles],
            "Link": [article.link for article in self.articles],
            "Content": [article.content for article in self.articles]
        })

    def identify_top_topics(self, num_topics=3):
        """Identify the top topics based on frequency and print meaningful summaries."""
        topics = self.topic_model.get_topic_freq()
        topics = topics[topics["Topic"] != -1]  # Remove outliers
        print(f"{len(topics)} topics identified")
        self.top_topics = topics.head(num_topics)
        print("\n🔹 Top Topics:\n")
        for topic in self.top_topics["Topic"]:
            topic_words = self.topic_model.get_topic(topic)
            topic_keywords = ", ".join([word[0] for word in topic_words[:8]])
            print(f"📌 **Topic {topic}:** {topic_keywords}")

            print("   🔹 Top Article Titles:")
            topic_articles = self.df[self.df["Topic"] == topic].head(15)

            for i, title in enumerate(topic_articles["Title"]):
                print(f"     {i + 1}. {title}")
            print("\n")

    def select_top_topics_by_score(self, num_topics=3):
        """Select top N topics using a composite score from penalized size and BERTScore."""

        print("📊 Selecting top topics using composite relevance score with size penalty...")

        topic_freq = self.topic_model.get_topic_freq()
        topic_freq = topic_freq[topic_freq["Topic"] != -1]  # Exclude outliers
        print(f"Number of topics identified: {len(topic_freq)}")

        # Compute BERTScore per topic
        bert_scores = {}
        for topic_id in topic_freq["Topic"]:
            rep_docs = self.topic_model.get_representative_docs(topic_id)
            if not rep_docs:
                continue
            keywords = ", ".join([word for word, _ in self.topic_model.get_topic(topic_id)[:8]])
            selected_docs = rep_docs[:5]
            scores = []
            for doc in selected_docs:
                _, _, F1 = score([keywords], [doc], lang="en", verbose=False)
                scores.append(F1.item())
            bert_scores[topic_id] = sum(scores) / len(scores) if scores else 0

        # Add BERTScore to DataFrame
        topic_freq["BERTScore"] = topic_freq["Topic"].map(bert_scores).fillna(0)

        # Apply log transform to topic size
        topic_freq["LogCount"] = topic_freq["Count"].apply(lambda x: np.log1p(x))

        # Normalize both metrics
        scaler = MinMaxScaler()
        topic_freq[["Size_Score", "BERT_Score_Norm"]] = scaler.fit_transform(
            topic_freq[["LogCount", "BERTScore"]]
        )

        # Penalize topics above size threshold
        size_threshold = topic_freq["Count"].quantile(0.75)
        topic_freq["Penalty"] = topic_freq["Count"].apply(lambda x: 0.7 if x > size_threshold else 1.0)
        topic_freq["Size_Score_Penalized"] = topic_freq["Size_Score"] * topic_freq["Penalty"]

        # Final relevance score with penalty applied
        topic_freq["RelevanceScore"] = (
                0.4 * topic_freq["Size_Score_Penalized"] + 0.6 * topic_freq["BERT_Score_Norm"]
        )

        # Sort and select top N
        topic_freq = topic_freq.sort_values("RelevanceScore", ascending=False)
        self.top_topics = topic_freq.head(num_topics)

        print("\n🔹 Selected Top Topics by Relevance Score (with penalty):")
        for topic, rel_score in zip(self.top_topics["Topic"], self.top_topics["RelevanceScore"]):
            keywords = ", ".join([word for word, _ in self.topic_model.get_topic(topic)[:6]])
            n_articles = topic_freq[topic_freq["Topic"] == topic]["Count"].values[0]
            print(f"📌 Topic {topic} — Size: {n_articles} — Score: {rel_score:.4f} — Keywords: {keywords}")
            print("   🔹 Top Article Titles:")
            topic_articles = self.df[self.df["Topic"] == topic].head(5)
            for i, title in enumerate(topic_articles["Title"]):
                print(f"     {i + 1}. {title}")

    def evaluate_all_metrics(self, evaluation_results_df: pd.DataFrame):
        """Evaluate all metrics and append results to a shared DataFrame."""
        print(f"📊 Running evaluation for: {self.week_value}")

        coherence = evaluate_coherence(self.topic_model, self.texts)
        diversity, redundancy = evaluate_diversity_redundancy(self.topic_model)
        bertscore = evaluate_topic_quality_with_bertscore(self.topic_model)

        row = {
            "Week": self.week_value,
            "Coherence": coherence,
            "Diversity": diversity,
            "Redundancy": redundancy,
            "BERTScore": bertscore
        }

        evaluation_results_df.loc[len(evaluation_results_df)] = row

    def save_articles_by_topic(self):
        """Group articles by top topics and save them in Markdown format for NotebookLM."""
        top_articles = self.df[self.df["Topic"].isin(self.top_topics["Topic"])]
        self.articles_by_topic = {
            topic: top_articles[top_articles["Topic"] == topic][["Title", "Link", "Content"]].to_dict(orient="records")
            for topic in self.top_topics["Topic"]
        }

        os.makedirs("topics", exist_ok=True)
        safe_week = self.week_value.replace("/", "-")

        for topic in self.top_topics["Topic"]:
            topic_words = self.topic_model.get_topic(topic)
            keywords = ", ".join([word[0] for word in topic_words[:8]])

            md_path = os.path.join("topics", f"{safe_week}_topic_{topic}.md")

            with open(md_path, "w", encoding="utf-8") as f:
                f.write(f"# Topic {topic}\n")
                f.write(
                    f"**The following articles were grouped together and are described by the words:** {keywords}\n\n")

                for article in self.articles_by_topic[topic][:15]:
                    f.write(f"## {article['Title']}\n")
                    f.write(f"**Link:** {str(article['Link'])}\n\n")
                    f.write(f"{article['Content']}\n\n")
                    f.write("---\n\n")

            print(f"✅ Saved Markdown: {md_path}")

    def visualize_topics(self, run_number):
        """Generate visualizations and save them to charts/ subfolder."""
        os.makedirs("charts", exist_ok=True)

        bar_chart = self.topic_model.visualize_barchart(top_n_topics=None)
        bar_chart.write_html(f"charts/bar_chart_run{run_number}.html")

        topics_vis = self.topic_model.visualize_topics()
        topics_vis.write_html(f"charts/topic_map_run{run_number}.html")

    def run_pipeline(self, evaluation_df: pd.DataFrame, save=True):
        """Execute the entire topic modeling pipeline."""
        self.load_articles()
        embeddings = self.generate_embeddings()
        self.fit_model(embeddings)
        self.select_top_topics_by_score()
        #self.evaluate_all_metrics(evaluation_df)
        if save:
            self.visualize_topics(run_number=12)
            #self.save_articles_by_topic()

# Execute the pipeline
if __name__ == "__main__":
    # week_value = ["10/02 - 16/02", "17/02 - 23/02", "24/02 - 02/03"]
    evaluation_df = pd.DataFrame(columns=[
        "Week", "Coherence", "Diversity", "Redundancy", "BERTScore"
    ])
    pipeline = TopicModelingPipeline("24/02 - 02/03")
    pipeline.run_pipeline(evaluation_df)

    # Running for several weeks at once:
    # weeks = ["10/02 - 16/02", "17/02 - 23/02", "24/02 - 02/03"]
    #
    # for week in weeks:
    #     pipeline = TopicModelingPipeline(week)
    #     pipeline.run_pipeline(evaluation_df, save=False)
    #
    # # Compute mean metrics
    # mean_row = evaluation_df.drop(columns=["Week"]).mean()
    # mean_row["Week"] = "Mean"
    # evaluation_df.loc[len(evaluation_df)] = mean_row
    #
    # # Print final table
    # print("\n📊 Evaluation Results Summary:")
    # print(evaluation_df)
