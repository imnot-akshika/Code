import asyncio
from itertools import count
import aiohttp
import json
import csv
import time
import re
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
from typing import Counter, Optional

@dataclass
class User:
    id: int
    name: str
    email: str
    city: str
    company: str

@dataclass  
class Post:
    id: int
    user_id: int
    title: str
    word_count: int

@dataclass
class Comment:
    id: int
    post_id: int
    email: str
    valid_email: bool

class DataFetcher:
    BASE_URL = "https://jsonplaceholder.typicode.com"
    EMAIL_RE = re.compile(r"^[\w.-]+@[\w.-]+\.\w{2,}$")

    def __init__(self, max_concurrent: int = 10):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.users: list[User] = []
        self.posts: list[Post] = []
        self.comments: list[Comment] = []
        self._fetch_times: dict[str, float] = {}

    async def fetch(self, session, url: str) -> dict:
        async with self.semaphore:
            try:
                async with session.get(url) as response:
                    return await response.json()
            except Exception:
                return None
            
    async def async_fetch_users(self, session) -> list[User]:
            data = await self.fetch(session, f"{self.BASE_URL}/users")
            if data:
                self.users = [User(id=u["id"], name=u["name"], email=u["email"], city=u["address"]["city"], company=u["company"]["name"]) for u in data]
            return []
        
    async def async_fetch_posts(self, session) -> list[Post]:
            data = await self.fetch(session, f"{self.BASE_URL}/posts")
            if data:
                self.posts = [Post(id=p["id"], user_id=p["userId"], title=p["title"], word_count=len(p["title"].split())) for p in data]
            return []
        
    async def async_fetch_comments(self, session) -> list[Comment]:
            data = await self.fetch(session, f"{self.BASE_URL}/comments")
            if data:
                self.comments = [Comment(id=c["id"], post_id=c["postId"], email=c["email"], valid_email=bool(self.EMAIL_RE.match(c["email"]))) for c in data]
            return []
        
    async def run(self) -> dict[str, float]:
        start_time = time.time()
        async with aiohttp.ClientSession() as session:
            await asyncio.gather(
                self.async_fetch_users(session),
                self.async_fetch_posts(session),
                self.async_fetch_comments(session)
            )
        self._fetch_times["total"] = time.time() - start_time
        return self._fetch_times
    
    def save_report(self, outfput_dir: Path) -> None:
        out = Path(outfput_dir)
        out.mkdir(exist_ok=True)

        with open(out / "users.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "name", "email", "city", "company"])
            writer.writeheader()
            writer.writerows(vars(u) for u in self.users)

        with open(out / "posts.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "user_id", "title", "word_count"])
            writer.writeheader()
            writer.writerows(vars(p) for p in self.posts)

        with open(out / "summary.json", "w") as f:
            json.dump(self.summary(), f, indent=2)

    def summary(self):
        from collections import Counter
        post_counts = Counter(p.user_id for p in self.posts)
        top_user_id = post_counts.most_common(1)[0][0]
        top_user = next(u for u in self.users if u.id == top_user_id)

        return {
    "fetched_at": datetime.now().isoformat(),
    "users": len(self.users),
    "posts": len(self.posts),
    "comments": len(self.comments),
    "invalid_emails": sum(1 for c in self.comments if not c.valid_email),
    "avg_post_length": round(sum(p.word_count for p in self.posts) / len(self.posts) if self.posts else 0, 2),
    "most_prolific_user": top_user.name,
    "fetch_times": self._fetch_times
}
    

#Example Usage
async def main():
    fetcher = DataFetcher(max_concurrent=10)
    
    start = time.perf_counter()
    timing = await fetcher.run()
    elapsed = time.perf_counter() - start
    
    print(f"Total time: {elapsed:.2f}s")
    print(f"Fetch times: {timing}")
    
    s = fetcher.summary()
    print(f"Users: {s['users']}, Posts: {s['posts']}, Comments: {s['comments']}")
    print(f"Invalid emails: {s['invalid_emails']}")
    print(f"Avg post length: {s['avg_post_length']} words")
    print(f"Most prolific user: {s['most_prolific_user']}")
    
    fetcher.save_report("output")
    print("Report saved to output/")

asyncio.run(main())