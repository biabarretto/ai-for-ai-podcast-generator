import requests
from bs4 import BeautifulSoup
from datetime import datetime
from googlesearch import search
import csv


def search_ai_news(sites, start_date, end_date, save_csv=False, csv_filename="news_results.csv"):
    all_results = []

    for site in sites:
        query = f"artificial intelligence site:{site} after:{start_date} before:{end_date}"
        print(f"Searching {site}...")
        try:
            links = search(query, num_results=10)  # You can adjust how many articles per site

            for link in links:
                try:
                    res = requests.get(link, timeout=10)
                    soup = BeautifulSoup(res.text, 'html.parser')
                    title = soup.title.string.strip() if soup.title else "No Title Found"
                    all_results.append({
                        "site": site,
                        "title": title,
                        "url": link
                    })
                except Exception as e:
                    print(f"Error fetching {link}: {e}")
        except Exception as e:
            print(f"Error searching {site}: {e}")

    if save_csv:
        week_label = f"{start_date}_to_{end_date}".replace("-", "")
        csv_filename = f"news_results_{week_label}.csv"
        keys = all_results[0].keys() if all_results else ["site", "title", "url"]
        with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"Results saved to {csv_filename}")

    return all_results


if __name__ == "__main__":
    sites = [
        "techcrunch.com",
        "the-decoder.com",
        "technologyreview.com"
    ]

    start_date = input("Enter start date (YYYY-MM-DD): ")
    end_date = input("Enter end date (YYYY-MM-DD): ")
    save_option = input("Save results to CSV? (yes/no): ").strip().lower()

    save_csv = save_option == "yes"
    news = search_ai_news(sites, start_date, end_date, save_csv)

    print("\nResults:")
    for idx, item in enumerate(news, start=1):
        print(f"{idx}. [{item['site']}] {item['title']}\n{item['url']}\n")
