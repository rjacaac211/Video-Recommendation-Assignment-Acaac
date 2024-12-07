import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://api.socialverseapp.com"
FLIC_TOKEN = os.environ.get("FLIC_TOKEN")

def fetch_paginated_data(endpoint, requires_auth=False, key="posts", extra_params=""):
    all_items = []
    page = 1
    headers = {"Flic-Token": FLIC_TOKEN} if requires_auth else {}

    while True:
        url = f"{BASE_URL}{endpoint}?page={page}&page_size=1000"
        if extra_params:
            url += f"&{extra_params}"

        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()

        items = data.get(key, [])
        if not items:
            break

        all_items.extend(items)
        page += 1

    return all_items

def save_data(filename, data):
    os.makedirs("data", exist_ok=True)
    filepath = os.path.join("data", filename)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

if __name__ == "__main__":
    # Fetch Viewed Posts
    viewed = fetch_paginated_data(
        endpoint="/posts/view",
        requires_auth=False,
        key="posts",
        extra_params="resonance_algorithm=resonance_algorithm_cjsvervb7dbhss8bdrj89s44jfjdbsjd0xnjkbvuire8zcjwerui3njfbvsujc5if"
    )
    save_data("viewed_posts.json", viewed)

    # Fetch Liked Posts
    liked = fetch_paginated_data(
        endpoint="/posts/like",
        requires_auth=False,
        key="posts",
        extra_params="resonance_algorithm=resonance_algorithm_cjsvervb7dbhss8bdrj89s44jfjdbsjd0xnjkbvuire8zcjwerui3njfbvsujc5if"
    )
    save_data("liked_posts.json", liked)

    # Fetch Inspired Posts
    inspired = fetch_paginated_data(
        endpoint="/posts/inspire",
        requires_auth=False,
        key="posts",
        extra_params="resonance_algorithm=resonance_algorithm_cjsvervb7dbhss8bdrj89s44jfjdbsjd0xnjkbvuire8zcjwerui3njfbvsujc5if"
    )
    save_data("inspired_posts.json", inspired)

    # Fetch Rated Posts
    rated = fetch_paginated_data(
        endpoint="/posts/rating",
        requires_auth=False,
        key="posts",
        extra_params="resonance_algorithm=resonance_algorithm_cjsvervb7dbhss8bdrj89s44jfjdbsjd0xnjkbvuire8zcjwerui3njfbvsujc5if"
    )
    save_data("rated_posts.json", rated)

    # Fetch All Posts (Requires Auth)
    all_posts = fetch_paginated_data(
        endpoint="/posts/summary/get",
        requires_auth=True,
        key="posts"
    )
    save_data("all_posts.json", all_posts)

    # Fetch All Users (Requires Auth)
    all_users = fetch_paginated_data(
        endpoint="/users/get_all",
        requires_auth=True,
        key="users"
    )
    save_data("all_users.json", all_users)
