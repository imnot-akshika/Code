import asyncio
import aiohttp
import json
import time
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Article:
    title: str
    source: str
    url: str
    summary: str = ""

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "source": self.source,
            "url": self.url,
            "summary": self.summary
        }

class AsyncAggregator:
    def __init__(self, max_concurrent: int = 5):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.articles: list[Article] = []

    async def fetch(self, session: aiohttp.ClientSession, url: str) -> dict | None:
        async with self.semaphore:
            try:
                async with session.get(url) as response:
                    return await response.json()
            except Exception:
                return None


    async def fetch_posts(self, session: aiohttp.ClientSession) -> list[Article]:
        async with self.semaphore:
            # fetch posts 1-10 from jsonplaceholder
            async with session.get("https://jsonplaceholder.typicode.com/posts") as response:
                posts = await response.json()
                return [Article(title=p["title"], source="jsonplaceholder", url=f"https://jsonplaceholder.typicode.com/posts/{p['id']}") for p in posts[:10]]

    async def fetch_users(self, session: aiohttp.ClientSession) -> list[Article]:
        async with self.semaphore:
            async with session.get("https://jsonplaceholder.typicode.com/users") as response:
                users = await response.json()
                return [Article(title=u["name"], source=u["company"]["name"], url=u["email"]) for u in users]


    async def run(self) -> dict:
        async with aiohttp.ClientSession() as session:
            posts, users = await asyncio.gather(
                self.fetch_posts(session),
                self.fetch_users(session)
            )
            self.articles = posts + users
            return self.summary()

    def save(self, filename: str) -> None:
        with open(filename, "w") as f:
            json.dump([a.to_dict() for a in self.articles], f, indent=2)

    def summary(self) -> dict:
        sources = {}
        for a in self.articles:
            sources[a.source] = sources.get(a.source, 0) + 1
        return {
            "total": len(self.articles),
            "by_source": sources
        }
    



#example usage
async def main():
    aggregator = AsyncAggregator(max_concurrent=5)
    
    start = time.perf_counter()
    result = await aggregator.run()
    elapsed = time.perf_counter() - start
    
    print(f"Fetched {result['total']} items in {elapsed:.2f}s")
    print(aggregator.summary())
    aggregator.save("articles.json")

asyncio.run(main())