from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
import pandas as pd


model = SentenceTransformer('all-MiniLM-L6-v2')

# Assume articles is a list of cleaned article texts
embeddings = model.encode(articles, show_progress_bar=True)

topic_model = BERTopic()
topics, probs = topic_model.fit_transform(articles, embeddings)

top_topics = topic_model.get_topic_freq().head(3)
print(top_topics)
for topic in top_topics['Topic']:
    print(f"Topic {topic}:")
    print(topic_model.get_topic(topic))  # Displays top words
    print(topic_model.get_representative_docs(topic))  # Retrieves top articles

topic_model.visualize_barchart(top_n_topics=3)
topic_model.visualize_topics()

df = pd.DataFrame({"Article": articles, "Topic": topics})
top_articles = df[df["Topic"].isin(top_topics["Topic"].tolist())]
