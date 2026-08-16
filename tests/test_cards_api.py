import os
import requests
import json
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))
API_TOKEN = os.getenv("TOKEN")
BASE_URL = "https://proxy.royaleapi.dev/v1"

def fetch_cards():
    if not API_TOKEN:
        print("Error: TOKEN not found in .env")
        return
        
    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
        "Accept": "application/json"
    }
    
    print("Fetching from API...")
    response = requests.get(f"{BASE_URL}/cards", headers=headers)
    if response.status_code == 200:
        data = response.json()
        print(f"Total items found: {len(data.get('items', []))}")
        
        # Print a sample of 3 items to inspect the structure
        sample = data.get("items", [])[:3]
        print("\nSample items:")
        print(json.dumps(sample, indent=2))
    else:
        print(f"Failed to fetch cards: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    fetch_cards()
