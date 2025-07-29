'''
Local test of the API created in main.py
'''

import requests

url = 'http://127.0.0.1:8000'
request = requests.get(url)
print(request.status_code)
print(request.text)


data = {
    "age": 37,
    "workclass": "Private",
    "fnlgt": 178356,
    "education": "HS-grad",
    "education-num": 10,
    "marital-status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}

post_url = url+'/data'
posts = requests.post(post_url, json=data)

print(posts.status_code)
print(posts.text)
