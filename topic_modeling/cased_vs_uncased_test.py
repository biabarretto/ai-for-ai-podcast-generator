import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from sklearn.metrics.pairwise import cosine_similarity

from utils import clean_text, compute_coherence_score

# Load dataset
df = pd.read_csv("training_data.csv")

# Combine title and description for embeddings
df["text"] = df["title"] + " " + df["description"]
df["text"] = df["text"].apply(clean_text)

# Extract texts
texts = df["text"].tolist()

# --------------------- Test Uncased Model --------------------- #
print("\n🚀 Testing Uncased Model (all-MiniLM-L6-v2)...")
model_uncased = SentenceTransformer('all-MiniLM-L6-v2')  # 384-dim

# Generate embeddings
embeddings_uncased = model_uncased.encode(texts, show_progress_bar=True)

# Fit BERTopic with uncased embeddings
topic_model_uncased = BERTopic()
topics_uncased, probs_uncased = topic_model_uncased.fit_transform(texts, embeddings_uncased)

# Save Uncased Model Topics
df["Topic_Uncased"] = topics_uncased

# Evaluate coherence for uncased model
coherence_score_uncased = compute_coherence_score(topic_model_uncased, texts)
print(f"Coherence Score (Uncased Model): {coherence_score_uncased:.4f}")

# --------------------- Test Cased Model --------------------- #
print("\n🚀 Testing Cased Model (bert-base-nli-mean-tokens)...")
model_cased = SentenceTransformer("sentence-transformers/bert-base-nli-mean-tokens")  # 768-dim

# Generate embeddings
embeddings_cased = model_cased.encode(texts, show_progress_bar=True)

# Fit BERTopic with cased embeddings
topic_model_cased = BERTopic()
topics_cased, probs_cased = topic_model_cased.fit_transform(texts, embeddings_cased)

# Save Cased Model Topics
df["Topic_Cased"] = topics_cased

# Evaluate coherence for cased model
coherence_score_cased = compute_coherence_score(topic_model_cased, texts)
print(f"Coherence Score (Cased Model): {coherence_score_cased:.4f}")

# --------------------- Compare Similarity Between Models --------------------- #
print("\n🚀 Comparing Topic Similarity Between Cased and Uncased Models...")

# Get topic embeddings
topic_embeds_uncased = topic_model_uncased.topic_embeddings_
topic_embeds_cased = topic_model_cased.topic_embeddings_

# Ensure same number of topics for comparison
min_topics = min(len(topic_embeds_uncased), len(topic_embeds_cased))
topic_embeds_uncased = topic_embeds_uncased[:min_topics]
topic_embeds_cased = topic_embeds_cased[:min_topics]

# Compute cosine similarity between topic embeddings
similarity = cosine_similarity(topic_embeds_uncased, topic_embeds_cased)
avg_similarity = np.mean(np.diag(similarity))

print(f"Average Topic Similarity Between Models: {avg_similarity:.4f}")

# --------------------- Print Summary --------------------- #
print("\n🚀 Summary:")
print(f"Coherence Score (Uncased Model): {coherence_score_uncased:.4f}")
print(f"Coherence Score (Cased Model): {coherence_score_cased:.4f}")
print(f"Average Topic Similarity Between Models: {avg_similarity:.4f}")

# --------------------- Recommendations --------------------- #
if coherence_score_cased > coherence_score_uncased:
    print("\n📌 Recommendation: Use the Cased Model (bert-base-nli-mean-tokens) for better topic coherence.")
else:
    print("\n📌 Recommendation: Use the Uncased Model (all-MiniLM-L6-v2) for faster performance and similar coherence.")
