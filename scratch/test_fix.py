import requests

url = "http://127.0.0.1:8000/structure"
payload = {
    "transcript": "There is a bug in the database connection logic. It fails with a timeout.",
    "mode": "debug",
    "session_id": "test-session-123"
}

try:
    response = requests.post(url, json=payload)
    print(f"Status Code: {response.status_code}")
    print("Response JSON:")
    print(response.json())
except Exception as e:
    print(f"Error: {e}")
