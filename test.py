import requests
import json

# Test endpoint: Get All Viewed Posts
test_url = "https://api.socialverseapp.com/posts/view?page=1&page_size=1000&resonance_algorithm=resonance_algorithm_cjsvervb7dbhss8bdrj89s44jfjdbsjd0xnjkbvuire8zcjwerui3njfbvsujc5if"

try:
    response = requests.get(test_url)
    response.raise_for_status()
    data = response.json()
    
    # Print the keys in the JSON response
    print("Response keys:", data.keys())
    
    # Check for 'posts' instead of 'results'
    if "posts" in data:
        print("Number of posts in this page:", len(data["posts"]))
        if len(data["posts"]) > 0:
            print("Sample post:", json.dumps(data["posts"][0], indent=2))
    else:
        print("No 'posts' key found in response.")
        
except requests.exceptions.HTTPError as e:
    print(f"HTTP error occurred: {e}")
except Exception as ex:
    print(f"An error occurred: {ex}")