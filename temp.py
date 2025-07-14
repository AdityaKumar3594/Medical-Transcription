import requests

api_key = "AIzaSyC-3Rv2cliPlj9UOcskogjzeq-S6aJBun4"
url = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={api_key}"

headers = {
    "Content-Type": "application/json"
}
data = {
    "contents": [
        {
            "parts": [
                {"text": "Hello Gemini"}
            ]
        }
    ]
}

response = requests.post(url, headers=headers, json=data)

if response.status_code == 200:
    print("✅ API key is valid!")
    print(response.json())
else:
    print("❌ API key is invalid or not authorized.")
    print(response.status_code, response.json())
