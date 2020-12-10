import requests
from io import  open

def test_keras():
   
    response = requests.post("http://localhost:80/v1/vision/deletemodel",data={"name":"professiontestkeras","admin_key":"Olafenwa"}).json()
    assert response["success"] == True


def test_tf():
  
    response = requests.post("http://localhost:80/v1/vision/deletemodel",data={"name":"professiontesttf","admin_key":"Olafenwa"}).json()
    assert response["success"] == True

def test_onnx():
   
    response = requests.post("http://localhost:80/v1/vision/deletemodel",data={"name":"professiontestonnx","admin_key":"Olafenwa"}).json()
    assert response["success"] == True


