import requests

url = 'http://127.0.0.1:5000/predict'

data = {
    'age': 24,
    'goals': 12,
    'assists': 5,
    'appearances': 28,
    'club_rank': 1,
    'position': 'Forward'
}

response = requests.post(url, json=data)
print(response.json())
