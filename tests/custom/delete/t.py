import requests
from io import  open

def test_keras():
   
    response = requests.post("http://localhost:80/v1/vision/deletemodel",data={"name":"professiontestkeras","admin_key":"Olafenwa"}).json()
    print(response)

test_keras()
