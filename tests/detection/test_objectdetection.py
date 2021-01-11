from io import open

import requests


def test_detection():

    image_data = open("detection.jpg", "rb").read()

    response = requests.post(
        "http://localhost:80/v1/vision/detection",
        files={"image": image_data},
        data={"api_key": "Olafenwa1"},
    ).json()

    assert response["success"] == True
    assert len(response["predictions"]) == 3
