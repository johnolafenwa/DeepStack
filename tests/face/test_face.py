import time
from io import open

import requests


def test_detection():

    image_data = open("detection.jpg", "rb").read()
    t1 = time.time()
    response = requests.post(
        "http://localhost:80/v1/vision/face", files={"image": image_data}
    ).json()
    print("Duration: ", time.time() - t1)
    assert response["success"] == True
    assert len(response["predictions"]) == 4


def test_register():

    image_data = open("adele1.jpg", "rb").read()
    requests.post(
        "http://localhost:80/v1/vision/face/register",
        files={"image": image_data},
        data={"userid": "Adele"},
    ).json()

    response = requests.post(
        "http://localhost:80/v1/vision/face/register",
        files={"image": image_data},
        data={"userid": "Adele"},
    ).json()

    assert response["success"] == True


def test_recognize():

    time.sleep(4)

    image_data = open("adele2.jpg", "rb").read()

    response = requests.post(
        "http://localhost:80/v1/vision/face/recognize", files={"image": image_data}
    ).json()

    assert response["success"] == True
    assert response["predictions"][0]["userid"] == "Adele"


def test_list():

    response = requests.post("http://localhost:80/v1/vision/face/list").json()

    assert response["success"] == True
    assert "Adele" in response["faces"]


def test_delete():

    time.sleep(6)

    image_data = open("adele1.jpg", "rb").read()

    response = requests.post(
        "http://localhost:80/v1/vision/face/delete", data={"userid": "Adele"}
    ).json()

    assert response["success"] == True


def test_recognize_after_delete():

    time.sleep(10)

    image_data = open("adele2.jpg", "rb").read()

    response = requests.post(
        "http://localhost:80/v1/vision/face/recognize", files={"image": image_data}
    ).json()

    assert response["success"] == True
    assert response["predictions"][0]["userid"] == "unknown"


def test_list_after_delete():

    time.sleep(10)

    response = requests.post("http://localhost:80/v1/vision/face/list").json()

    assert response["success"] == True
    assert "Adele" not in response["faces"]
