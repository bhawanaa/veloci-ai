import os
import requests
import certifi

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("OPENAI_API_KEY not set!")
    exit(1)

url = "https://api.openai.com/v1/models"
headers = {
    "Authorization": f"Bearer {OPENAI_API_KEY}",
}

try:
    response = requests.get(url, headers=headers, verify=False, timeout=10)
    print("Status code:", response.status_code)
    print("Response:", response.text[:500])
except Exception as e:
    print("Connection error:", e)
