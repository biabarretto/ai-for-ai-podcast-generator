import sqlite3

# Note: change the path to where your data is stored locally
DB_PATH = r"G:\My Drive\scraped_articles.db"


def create_table():
    """Create the database table if it doesn't exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS articles (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source TEXT,
        link TEXT UNIQUE,
        title TEXT,
        category TEXT, 
        description TEXT, 
        content TEXT,
        pub_date TEXT,
        scraped_date TEXT, 
        week TEXT
    )
    """)

    conn.commit()
    conn.close()


# Run this once to ensure the table is created
if __name__ == "__main__":
    create_table()
