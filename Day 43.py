import requests
import json
from pathlib import Path
from requests.exceptions import HTTPError, ConnectionError, Timeout

class APIClient:
    BASE_URL = "https://jsonplaceholder.typicode.com"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def get_users(self) -> list[dict]:
        response = self.session.get(f"{self.BASE_URL}/users")
        response.raise_for_status()
        return response.json()

    def get_user(self, user_id: int) -> dict | None:
        response = self.session.get(f"{self.BASE_URL}/users/{user_id}")
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()

    def get_user_posts(self, user_id: int) -> list[dict]:
        response = self.session.get(f"{self.BASE_URL}/posts", params={"userId": user_id})
        response.raise_for_status()
        return response.json()

    def create_post(self, title: str, body: str, user_id: int) -> dict | None:
        response = self.session.post(f"{self.BASE_URL}/posts", json={
            "title": title,
            "body": body,
            "userId": user_id
        })
        response.raise_for_status()
        return response.json()

    def save_to_json(self, data: any, filename: str) -> None:
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)

    def summary(self) -> dict:
        users = self.get_users()
        posts = [post for user in users for post in self.get_user_posts(user["id"])]
        total_users = len(users)
        total_posts = len(posts)
        most_active_user = max(users, key=lambda user: len(self.get_user_posts(user["id"])))
        return {
            "total_users": total_users,
            "total_posts": total_posts,
            "most_active": most_active_user["username"]
        }
    


#Example usage

client = APIClient()

users = client.get_users()
print(f"Total users: {len(users)}")

user = client.get_user(4)
print(f"User 1: {user['name']}")

posts = client.get_user_posts(1)
print(f"User 1 posts: {len(posts)}")

new_post = client.create_post("My Title", "My body text", 1)
print(f"Created: {new_post}")

client.save_to_json(users, "users.json")
print(client.summary())