import os
import json
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
BASE_URL = "https://api.socialverseapp.com"
FLIC_TOKEN = os.environ.get("FLIC_TOKEN")
DATA_DIR = "data"

def fetch_paginated_data(endpoint, requires_auth=False, key="posts", extra_params=""):
    """
    Fetches paginated data from the given API endpoint.
    
    Args:
        endpoint (str): API endpoint (relative to the BASE_URL).
        requires_auth (bool): Whether authentication is required for the endpoint.
        key (str): Key to extract specific data from the API response.
        extra_params (str): Additional query parameters.

    Returns:
        list: List of items retrieved from the API.
    """
    all_items = []
    page = 1
    headers = {"Flic-Token": FLIC_TOKEN} if requires_auth else {}

    while True:
        url = f"{BASE_URL}{endpoint}?page={page}&page_size=1000"
        if extra_params:
            url += f"&{extra_params}"

        print(f"Fetching data from: {url}")
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()

        items = data.get(key, [])
        if not items:
            break

        all_items.extend(items)
        page += 1

    print(f"Fetched {len(all_items)} items from {endpoint}")
    return all_items

def save_data(filename, data):
    """
    Saves data to a JSON file.
    
    Args:
        filename (str): Name of the file to save the data.
        data (list): Data to be saved.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    filepath = os.path.join(DATA_DIR, filename)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Data saved to {filepath}")

def fetch_and_save_all():
    """
    Fetches all required data from the APIs and saves them to local files.
    """
    api_endpoints = {
        "viewed_posts.json": {
            "endpoint": "/posts/view",
            "requires_auth": False,
            "key": "posts",
            "extra_params": "resonance_algorithm=resonance_algorithm_cjsvervb7dbhss8bdrj89s44jfjdbsjd0xnjkbvuire8zcjwerui3njfbvsujc5if"
        },
        "liked_posts.json": {
            "endpoint": "/posts/like",
            "requires_auth": False,
            "key": "posts",
            "extra_params": "resonance_algorithm=resonance_algorithm_cjsvervb7dbhss8bdrj89s44jfjdbsjd0xnjkbvuire8zcjwerui3njfbvsujc5if"
        },
        "inspired_posts.json": {
            "endpoint": "/posts/inspire",
            "requires_auth": False,
            "key": "posts",
            "extra_params": "resonance_algorithm=resonance_algorithm_cjsvervb7dbhss8bdrj89s44jfjdbsjd0xnjkbvuire8zcjwerui3njfbvsujc5if"
        },
        "rated_posts.json": {
            "endpoint": "/posts/rating",
            "requires_auth": False,
            "key": "posts",
            "extra_params": "resonance_algorithm=resonance_algorithm_cjsvervb7dbhss8bdrj89s44jfjdbsjd0xnjkbvuire8zcjwerui3njfbvsujc5if"
        },
        "all_posts.json": {
            "endpoint": "/posts/summary/get",
            "requires_auth": True,
            "key": "posts",
            "extra_params": ""
        },
        "all_users.json": {
            "endpoint": "/users/get_all",
            "requires_auth": True,
            "key": "users",
            "extra_params": ""
        }
    }

    for filename, config in api_endpoints.items():
        data = fetch_paginated_data(
            endpoint=config["endpoint"],
            requires_auth=config["requires_auth"],
            key=config["key"],
            extra_params=config["extra_params"]
        )
        save_data(filename, data)

if __name__ == "__main__":
    fetch_and_save_all()
