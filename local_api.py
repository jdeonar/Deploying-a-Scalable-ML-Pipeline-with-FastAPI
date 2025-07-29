import json

import requests

r = 'http://127.0.0.1:8000'
request = requests.get(r)
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

post_r = r+'/data'
posts = requests.post(post_r, json=data)

print(posts.status_code)
print(posts.text)
