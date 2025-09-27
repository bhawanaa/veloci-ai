import requests

# Change these to match your server address and port
API_URL = "http://localhost:8080/ask"

def ask_about_document(question):
    response = requests.post(API_URL, data={"question": question})
    print("Status:", response.status_code)
    print("Response:", response.json())

if __name__ == "__main__":
    ask_about_document("whats inside the document?")
