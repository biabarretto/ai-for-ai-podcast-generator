import sqlite3
from datetime import datetime

from data_model.database import DB_PATH
from data_model.models import ScrapedArticle
import numpy as np

import re
import unicodedata
import nltk
from nltk.corpus import stopwords

from gensim.corpora.dictionary import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from bertopic import BERTopic
import numpy as np
import random

# Ensure stopwords are downloaded
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


def clean_text(text):
    """Cleans text: removes markdown, punctuation, stopwords, and normalizes spaces."""
    if not isinstance(text, str):
        return ""

    # Remove markdown links
    text = re.sub(r'\[.*?\]\(.*?\)', '', text)
    # Normalize unicode characters (e.g., accents)
    text = unicodedata.normalize("NFKD", text)
    # Remove punctuation (but keep words and spaces)
    text = re.sub(r'[^\w\s]', '', text)

    return text

def clean_plain_text(text):
    """
    Cleans plain text (like article titles) to match cleaned HTML content.
    """
    clean_text = re.sub(r'\s+', ' ', text)                    # Normalize whitespace
    clean_text = re.sub(r'&[a-z]+;', '', clean_text)          # Remove HTML entities
    clean_text = re.sub(r'[^\w\s.,]', '', clean_text)         # Keep only letters, digits, spaces, periods, and commas
    return clean_text.strip()


def get_articles(week_values):
    """Retrieve articles from the database for one or multiple weeks.

    Args:
        week_values (str or list): a single week string or a list of such strings.
    Returns:
        articles: list of all ScrapedArticle objects for the given weeks
        texts: cleaned texts (title + description) for topic modeling
    """
    if isinstance(week_values, str):
        week_values = [week_values]

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    placeholders = ",".join(["?"] * len(week_values))
    query = f"""
        SELECT source, link, title, category, description, content, pub_date, scraped_date, week
        FROM articles
        WHERE week IN ({placeholders})
    """
    cursor.execute(query, week_values)
    rows = cursor.fetchall()
    conn.close()

    articles = []
    texts = []

    for row in rows:
        article = ScrapedArticle(
            source=row[0],
            link=row[1],
            title=row[2],
            category=row[3].split(", "),
            description=row[4],
            content=row[5],
            pub_date=datetime.fromisoformat(row[6]),
            scraped_date=datetime.fromisoformat(row[7]),
            week=row[8]
        )
        articles.append(article)

        full_text = f"{clean_plain_text(article.title)} {article.description}"
        marker = f"The post {clean_plain_text(article.title)}"
        cut_text = full_text.split(marker)[0].strip()

        texts.append(clean_text(cut_text))

    return articles, texts


def compute_coherence_score(topic_model, texts):
    """Computes coherence score using c_v metric from Octis."""
    # Get topics and their keywords
    topics = topic_model.get_topics()
    topic_words = [[word for word, _ in topic_model.get_topic(topic)] for topic in topics]

    # Ensure texts are tokenized (list of words, not raw strings)
    tokenized_texts = [text.split() for text in texts]

    # Compute coherence using Octis
    coherence = Coherence(texts=tokenized_texts, measure="c_v")
    coherence_score = coherence.score({"topics": topic_words})  # Wrap topics in a dictionary

    return np.mean(coherence_score)  # Return the average coherence score


def evaluate_coherence(topic_model: BERTopic, texts: list) -> float:
    """Calculate topic coherence using Gensim (C_v)."""
    topics_words = [
        [word for word, _ in topic_model.get_topic(topic)]
        for topic in topic_model.get_topics().keys() if topic != -1
    ]
    tokenized_texts = [text.lower().split() for text in texts]
    dictionary = Dictionary(tokenized_texts)
    corpus = [dictionary.doc2bow(text) for text in tokenized_texts]

    coherence_model = CoherenceModel(
        topics=topics_words,
        texts=tokenized_texts,
        corpus=corpus,
        dictionary=dictionary,
        coherence='c_v'
    )
    score = coherence_model.get_coherence()
    print(f"🔍 Coherence Score (C_v): {score:.4f}")
    return score


def evaluate_diversity_redundancy(topic_model: BERTopic, top_n=10) -> tuple:
    """Evaluate topic diversity and average redundancy between topics."""
    topics = [topic_model.get_topic(topic) for topic in topic_model.get_topics().keys() if topic != -1]
    topic_keywords = [[word for word, _ in topic[:top_n]] for topic in topics]

    all_keywords = sum(topic_keywords, [])
    unique_keywords = set(all_keywords)
    diversity = len(unique_keywords) / (len(topics) * top_n)

    vectorizer = CountVectorizer()
    topic_texts = [" ".join(words) for words in topic_keywords]
    topic_vectors = vectorizer.fit_transform(topic_texts).toarray()
    sim_matrix = cosine_similarity(topic_vectors)
    np.fill_diagonal(sim_matrix, 0)
    redundancy = sim_matrix.mean()

    print(f"🧩 Topic Diversity: {diversity:.4f}")
    print(f"🔁 Avg Redundancy: {redundancy:.4f}")
    return diversity, redundancy


def evaluate_stability(topic_model: BERTopic, texts: list, model, reruns=3, sample_size=100) -> float:
    """Evaluate model stability across multiple reruns with sampled data."""
    original_topics = topic_model.get_topics()
    base_sets = [set([word for word, _ in topic[:10]]) for topic in original_topics.values() if topic]
    jaccard_scores = []

    for _ in range(reruns):
        sample_indices = random.sample(range(len(texts)), min(sample_size, len(texts)))
        sample_texts = [texts[i] for i in sample_indices]
        sample_embeddings = model.encode(sample_texts, show_progress_bar=False)
        tmp_model = BERTopic(embedding_model=model)
        tmp_model.fit(sample_texts, sample_embeddings)

        sampled_sets = [set([word for word, _ in topic[:10]]) for topic in tmp_model.get_topics().values() if topic]

        for set1 in base_sets:
            scores = [len(set1 & set2) / len(set1 | set2) for set2 in sampled_sets if len(set1 | set2) > 0]
            if scores:
                jaccard_scores.append(max(scores))

    stability = np.mean(jaccard_scores)
    print(f"📈 Avg Topic Stability (Jaccard): {stability:.4f}")
    return stability
