# AI-for-AI Podcast Generator
This project leverages modern NLP and automation techniques to generate weekly podcast episodes summarizing the top 3 
trending topics in AI and ML. It is designed to help professionals in the field stay informed in this era of fast-paced developments, 
using tools like web scraping, topic modeling, and audio synthesis through Google’s NotebookLM.

The code provided in this repo collects articles published in AI magazines and journals through their RSS feeds. 
It then uses topic modeling to identify the top 3 topics of the week and generate a markdown file containing all articles
per topic, which can then be uploaded into NotebookLM to generate the final podcast. 
Although the current implementation focuses on AI articles, this code can be used for any chosen field or subject, simply
by altering the RSS feeds used to collect data. 

## 📁 Project Structure
data_model/: Scripts with database structure and pydantic model to store scraped articles.

scraper/: Code to scrape and preprocess data from various sources and save it in the database.

topic_modeling/: Code for BERTopic.

archive/: Additional scripts used but that are not essential to the main pipeline (attempt to finetune embeddings, script to collect articles from other sources to compare with podcast output)

README.md: Project documentation.

LICENSE: MIT License

## 🚀 How to Run

Follow the steps below to set up and run the **AI for AI** pipeline from start to finish.

### 1. Install Dependencies

Ensure [Poetry](https://python-poetry.org/docs/#installation) is installed on your system, then install project dependencies:

```bash
poetry install
```

### 2. Activate the Virtual Environment

```bash
poetry shell
```

### 3. Create the Scraped Articles Database

Set up the SQLite database where the scraped articles will be stored. 
Remember to change the DB_PATH variable to a valid path where you want the table to be stored:

```bash
python data_model/database.py
```

> 💡 This will initialize the `articles` table inside the SQLite database.

### 4. Scrape Articles

Fetch new articles from supported RSS feeds and store them in the database:

```bash
python scraping/rss_scraper.py 
```

> You can alter the RSS feeds directly in the 'rss_scraper.py' file. 
Only articles published in the last 2 days will be collected, so the script has to run daily 
in order to collect all articles for a given week. The database will automatically
> discard any duplicated entries, so you can run it at any time.

### 5. Run Topic Modeling

Use BERTopic to identify trending AI topics for the week. This script processes the titles and descriptions and clusters them using transformer-based embeddings:

```bash
python topic_modeling/bert_topic.py
```

Topics are scored and ranked to select the top 3 for summarization.

For each topic, a markdown file will be saved in the `topics/` folder, under the `topic_modeling/` folder. 
They can be directly uploaded into Google NotebookLM for podcast generation.


### 6. Upload to NotebookLM

Visit [notebooklm.google.com](https://notebooklm.google.com/), upload the three markdown files, and use this instruction template to guide podcast creation:

```
Generate a podcast titled "AI This Week" (mention the name) that summarizes the top 3 AI topics of the week. Each document includes ~15 articles grouped by theme. For each topic, synthesize the key insights across articles, emphasize trends, implications, and context, and avoid summarizing single pieces in isolation. Use the keywords at the top as a guide. The tone should be technical and analytical, targeting an audience familiar with AI and recent developments.
```

NotebookLM will generate a conversational and informative podcast-style summary hosted by two AI narrators.

## 📊 Dataset Schema
Due to data-sharing restrictions, the dataset cannot be publicly released. 
However, here is a mock sample of the schema used for storage and processing for reference:

```json
[
  {
    "source": "MarkTechPost",
    "link": "https://marktechpost.com/deepmind-gemini-2",
    "title": "DeepMind Releases Gemini 2: A New Era of Multi-Modal AI",
    "category": "AI Research",
    "description": "The new model outperforms GPT-4 in multiple benchmarks and introduces improvements in vision-language integration.",
    "content": "DeepMind has unveiled Gemini 2, a multi-modal model designed to bridge vision and language understanding. Early benchmarks show a significant improvement over GPT-4 in both zero-shot and fine-tuned settings. The model leverages a novel routing technique in its mixture-of-experts architecture...",
    "pub_date": "2025-04-20T08:00:00Z",
    "scraped_date": "2025-04-20T00:00:00",
    "week": "15/04 - 21/04"
  },
  {
    "source": "Unite.AI",
    "link": "https://unite.ai/llm-drug-discovery",
    "title": "Using LLMs to Simulate Drug Discovery Pipelines",
    "category": "Biotech",
    "description": "LLMs are being adapted to simulate and optimize drug development, particularly for molecular binding prediction.",
    "content": "A team at Stanford proposed a system that integrates LLMs with molecular dynamics simulations. The aim is to reduce computation time and improve prediction accuracy for protein-ligand binding affinity. Their pipeline uses fine-tuned transformers to estimate binding probabilities before full simulations...",
    "pub_date": "2025-04-19T10:30:00Z",
    "scraped_date": "2025-04-19T00:00:00",
    "week": "15/04 - 21/04"
  },
  {
    "source": "MIT News",
    "link": "https://news.mit.edu/2025/ai-autonomous-software-safety",
    "title": "MIT Researchers Develop New AI Safety Protocols for Autonomous Systems",
    "category": "Safety",
    "description": "A new probabilistic safety protocol aims to better detect and avoid failure modes in self-driving car algorithms.",
    "content": "MIT CSAIL researchers have introduced a framework called ProbSafeNet, which applies uncertainty-aware modeling to identify high-risk behaviors in real-time. The study shows its ability to reduce collision rates in simulation by 68% compared to existing baselines. This represents a significant step for the trustworthiness of AI-driven autonomy...",
    "pub_date": "2025-04-18T13:00:00Z",
    "scraped_date": "2025-04-1800:00:00",
    "week": "15/04 - 21/04"
  }
]
```

## ⚙️ Configuration

- **Database Path**: Set in `data_model/database.py`.
- **RSS Feeds**: Can be modified in `scraper/rss_scraper.py` when defining the rss_urls variable to instantiate the RSSScraper object.
- **Embedding Model**: Default is `all-mpnet-base-v2` in `topic_modeling/bert_topic.py`, can be changed by passing a different model to the embedding_model variable when instantiating the TopicModelingPipeline object.
- **Custom Stop Words**: Defined in the init method of the TopicModelingPipeline class, useful for filtering domain-specific noise.

