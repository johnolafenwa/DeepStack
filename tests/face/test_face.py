import time
from io import open
import os
import requests

IMAGES_DIR = os.getenv("TEST_IMAGES_DIR")
DEEPSTACK_URL = os.getenv("TEST_DEEPSTACK_URL")
API_KEY = os.getenv("TEST_API_KEY")

def test_detection():

    image_data = open(os.path.join(IMAGES_DIR,"face_detection.jpg"), "rb").read()
    response = requests.post(
        DEEPSTACK_URL+"/v1/vision/face", files={"image": image_data}, data={"api_key": API_KEY}
    ).json()
    assert response["success"] == True
    assert len(response["predictions"]) == 4


def test_register():

    image_data = open(os.path.join(IMAGES_DIR,"adele1.jpg"), "rb").read()
    requests.post(
        DEEPSTACK_URL+"/v1/vision/face/register",
        files={"image": image_data},
        data={"userid": "Adele","api_key": API_KEY},
    ).json()

    response = requests.post(
        DEEPSTACK_URL+"/v1/vision/face/register",
        files={"image": image_data},
        data={"userid": "Adele","api_key": API_KEY},
    ).json()

    assert response["success"] == True


def test_recognize():

    time.sleep(4)

    image_data = open(os.path.join(IMAGES_DIR,"adele2.jpg"), "rb").read()

    response = requests.post(
        DEEPSTACK_URL+"/v1/vision/face/recognize", files={"image": image_data}, data={"api_key": API_KEY}
    ).json()

    assert response["success"] == True
    assert response["predictions"][0]["userid"] == "Adele"


def test_list():

    response = requests.post(DEEPSTACK_URL+"/v1/vision/face/list",data={"api_key": API_KEY}).json()

    assert response["success"] == True
    assert "Adele" in response["faces"]


def test_delete():

    time.sleep(6)

    response = requests.post(
        DEEPSTACK_URL+"/v1/vision/face/delete", data={"userid": "Adele"}
    ).json()

    assert response["success"] == True


def test_recognize_after_delete():

    time.sleep(10)

    image_data = open(os.path.join(IMAGES_DIR,"adele2.jpg"), "rb").read()

    response = requests.post(
        DEEPSTACK_URL+"/v1/vision/face/recognize", files={"image": image_data}, data={"api_key": API_KEY}
    ).json()

    assert response["success"] == True
    assert response["predictions"][0]["userid"] == "unknown"


def test_list_after_delete():

    time.sleep(10)

    response = requests.post(DEEPSTACK_URL+"/v1/vision/face/list",data={"api_key": API_KEY}).json()

    assert response["success"] == True
    assert "Adele" not in response["faces"]
