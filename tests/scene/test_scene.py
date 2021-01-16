from io import open

import requests


def test_scene():

    image_data = open("scene.jpg", "rb").read()

    response = requests.post(
        "http://localhost:80/v1/vision/scene",
        files={"image": image_data}
    ).json()

    assert response["success"] == True
    assert response["label"] == "conference_room"
