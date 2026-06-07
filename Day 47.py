from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import threading
import time
import requests

class ConcurrentProcessor:
    def __init__(self, max_workers: int = 5):
        self.max_workers = max_workers
        self._lock = threading.Lock()
        self._results = []

    def process_files(self, directory: str, func) -> list:
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for file in Path(directory).glob("*.txt"):
                futures.append(executor.submit(func, file))
            for future in as_completed(futures):
                results.append(future.result())
        return results

    def fetch_urls(self, url: list[str]) -> dict:

        try:
            response = requests.get(url, timeout=10)
            return {"url": url, "status": response.status_code, "length": len(response.content)}
        except Exception as e:
            return {"url": url, "error": str(e)}


    def word_count_files(self, directory: str) -> dict[str, int]:
        def count(file):
            return file.name, len(file.read_text().split())
        
        results = {}
        with ThreadPoolExecutor (max_workers = self.max_workers) as executor:
            futures = [executor.submit(count, f) for f in Path(directory).glob("*.txt")]
            for future in as_completed(futures):
                filename, count = future.result()
                results[filename] = count
        return results
    


#Example Usage
# create some test files
import os
Path("test_files").mkdir(exist_ok=True)
for i in range(5):
    Path(f"test_files/file{i}.txt").write_text(
        f"This is file {i}. " * (i + 1) * 10
    )

processor = ConcurrentProcessor(max_workers=5)

# test word count
counts = processor.word_count_files("test_files")
for filename, count in counts.items():
    print(f"{filename}: {count} words")

# test URL fetching
urls = [
    "https://httpbin.org/delay/1",
    "https://httpbin.org/delay/1",
    "https://httpbin.org/delay/1",
]

start = time.perf_counter()
results = processor.fetch_urls(urls)
elapsed = time.perf_counter() - start

for r in results:
    print(r)
print(f"Fetched {len(urls)} URLs in {elapsed:.2f}s")