import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from bert_topic import TopicModelingPipeline


if __name__ == "__main__":
    # Instantiate 2 BERTopic objects, one for each embedding model
    week_value = "17/02 - 23/02"
    mpnet = TopicModelingPipeline(week_value)
    minilm = TopicModelingPipeline(week_value, embedding_model='all-MiniLM-L6-v2')

    # Get embeddings, fit model, evaluate
    print("\n Fitting Mpnet model \n")
    mpnet.load_articles()
    mpnet_embeddings = mpnet.generate_embeddings()
    mpnet.fit_model(mpnet_embeddings)
    mpnet.identify_top_topics()
    mpnet.evaluate_model()

    print("\n Fitting MiniLM model \n")
    minilm.load_articles()
    minilm_embeddings = minilm.generate_embeddings()
    minilm.fit_model(minilm_embeddings)
    minilm.identify_top_topics()
    minilm.evaluate_model()

    # Compare overlap in top topic keywords
    topic_keywords_mpnet = set(word for topic in mpnet.top_topics["Topic"]
                               for word, _ in mpnet.topic_model.get_topic(topic))
    topic_keywords_minilm = set(word for topic in minilm.top_topics["Topic"]
                                for word, _ in minilm.topic_model.get_topic(topic))

    overlap = topic_keywords_mpnet & topic_keywords_minilm
    print(f"🔁 Overlapping keywords between top topics: {len(overlap)}")
    print(overlap)
