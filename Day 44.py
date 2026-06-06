import requests
from bs4 import BeautifulSoup
import csv
import time
from dataclasses import dataclass

@dataclass
class Book:
    title: str
    price: float
    rating: int       # 1-5
    available: bool
    category: str = ""

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "price": self.price,
            "rating": self.rating,
            "available": self.available,
            "category": self.category
        }

class BookScraper:
    BASE_URL = "https://books.toscrape.com"
    RATING_MAP = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "BookScraper/1.0 (educational)"})
        self.books: list[Book] = []

    def scrape_page(self, url: str) -> list[Book]:
        try:
            response = self.session.get(url, timeout=10)
            response.encoding = "utf-8"
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"Error fetching {url}: {e}")
            return []
        
        soup = BeautifulSoup(response.text, "html.parser")
        books = []

        for article in soup.find_all("article", class_="product_pod"):
            try:
                title  = article.h3.a["title"]
                price = float(article.find("p", class_ ="price_color").text.replace("£", "").strip())
                rating_class = article.find("p", class_="star-rating")["class"][1]
                rating_word = rating_class
                rating = self.RATING_MAP[rating_word]
                availability_text = article.find("p", class_="availability").text.strip()
                available = "In stock" in availability_text
                books.append(Book(title, price, rating, available))

            except (AttributeError, ValueError, KeyError) as e:
                print(f"Error parsing book: {e}")
                continue

        return books


    def scrape_all(self, max_pages: int = 3) -> list[Book]:
        for page_num in range(1, max_pages + 1):
            if page_num == 1:
                url = f"{self.BASE_URL}/catalogue/"
            else:
                url = f"{self.BASE_URL}/catalogue/page-{page_num}.html"
        
            print(f"Scraping page {page_num}...")
            page_books = self.scrape_page(url)
            self.books.extend(page_books)
        
            time.sleep(1)  # Be polite and avoid hammering the server
    
        return self.books


    def filter_by_rating(self, min_rating: int) -> list[Book]:
        return [b for b in self.books if b.rating >= min_rating]

    def filter_by_price(self, max_price: float) -> list[Book]:
        return [b for b in self.books if b.price <= max_price]

    def save_to_csv(self, filename: str) -> None:
        if not self.books:
            return
        with open(filename, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.books[0].to_dict().keys())
            writer.writeheader()
            writer.writerows(b.to_dict() for b in self.books)

    def summary(self) -> dict:
        if not self.books:
            return {}
        prices = [b.price for b in self.books]
        return {
            "total": len(self.books),
            "avg_price": round(sum(prices) / len(prices), 2),
            "cheapest": min(prices),
            "most_expensive": max(prices),
            "five_star": len([b for b in self.books if b.rating == 5])
        }
    

#Example usage
scraper = BookScraper()
scraper.scrape_all(max_pages=2)

print(scraper.summary())
print(f"Books rated 4+: {len(scraper.filter_by_rating(4))}")
print(f"Books under £15: {len(scraper.filter_by_price(15.0))}")
scraper.save_to_csv("books.csv")