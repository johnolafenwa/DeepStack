import requests
from io import open

def test_scene():

    image_data = open("scene.jpg","rb").read()

    response = requests.post("http://localhost:80/v1/vision/scene",files={"image":image_data}, data={"api_key":"Mojohn1"}).json()

    assert response["success"] == True
    assert response["label"] == "conference_room"


