import requests


def test_keras():
    import requests

    image_data = open("test-image.jpg","rb").read()

    response = requests.post("http://localhost:80/v1/vision/custom/professiontestkeras",files={"image":image_data},data={"api_key":"Olafenwa1"}).json()
    assert response["success"] == True
    assert response["label"] == "farmer"


def test_tf():
    import requests

    image_data = open("test-image.jpg","rb").read()

    response = requests.post("http://localhost:80/v1/vision/custom/professiontesttf",files={"image":image_data},data={"api_key":"Olafenwa1"}).json()
    assert response["success"] == True
    assert response["label"] == "farmer"

def test_onnx():
    import requests

    image_data = open("test-image.jpg","rb").read()

    response = requests.post("http://localhost:80/v1/vision/custom/professiontestonnx",files={"image":image_data},data={"api_key":"Olafenwa1"}).json()
    assert response["success"] == True
    assert response["label"] == "farmer"

