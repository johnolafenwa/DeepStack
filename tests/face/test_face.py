import time
from io import open
import os
import requests

DATA_DIR = os.getenv("TEST_DATA_DIR")
DEEPSTACK_URL = os.getenv("TEST_DEEPSTACK_URL")
API_KEY = os.getenv("TEST_API_KEY")


def test_detection():

    image_data = open(os.path.join(
        DATA_DIR, "face_detection.jpg"), "rb").read()
    response = requests.post(
        DEEPSTACK_URL+"/v1/vision/face", files={"image": image_data}, data={"api_key": API_KEY}
    )
    response_json = response.json()

    assert response.status_code == 200, "Request failed with error: {}".format(
        response_json["error"])

    assert response_json["success"] == True
    assert len(response_json["predictions"]) == 4


def test_register():

    image_data = open(os.path.join(DATA_DIR, "adele1.jpg"), "rb").read()
    requests.post(
        DEEPSTACK_URL+"/v1/vision/face/register",
        files={"image": image_data},
        data={"userid": "Adele", "api_key": API_KEY},
    ).json()

    response = requests.post(
        DEEPSTACK_URL+"/v1/vision/face/register",
        files={"image": image_data},
        data={"userid": "Adele", "api_key": API_KEY},
    )

    response_json = response.json()

    assert response.status_code == 200, "Request failed with error: {}".format(
        response_json["error"])
    assert response_json["success"] == True


def test_recognize():

    time.sleep(4)

    image_data = open(os.path.join(DATA_DIR, "adele2.jpg"), "rb").read()

    response = requests.post(
        DEEPSTACK_URL+"/v1/vision/face/recognize", files={"image": image_data}, data={"api_key": API_KEY}
    )

    response_json = response.json()

    assert response.status_code == 200, "Request failed with error: {}".format(
        response_json["error"])
    assert response_json["success"] == True
    assert response_json["predictions"][0]["userid"] == "Adele"

    image_data = open(os.path.join(DATA_DIR, "obama1.jpg"), "rb").read()

    response = requests.post(
        DEEPSTACK_URL+"/v1/vision/face/recognize", files={"image": image_data}, data={"api_key": API_KEY}
    )

    response_json = response.json()

    assert response.status_code == 200, "Request failed with error: {}".format(
        response_json["error"])
    assert response_json["success"] == True
    assert response_json["predictions"][0]["userid"] == "unknown"


def test_match():

    time.sleep(4)

    image_data1 = open(os.path.join(DATA_DIR, "obama1.jpg"), "rb").read()
    image_data2 = open(os.path.join(DATA_DIR, "obama2.jpg"), "rb").read()
    image_data3 = open(os.path.join(DATA_DIR, "bradley.jpg"), "rb").read()

    response = requests.post(
        DEEPSTACK_URL+"/v1/vision/face/match", files={"image1": image_data1, "image2": image_data2}, data={"api_key": API_KEY}
    )

    response_json = response.json()

    assert response.status_code == 200, "Request failed with error: {}".format(
        response_json["error"])
    assert response_json["success"] == True
    assert response_json["similarity"] > 0.65

    response = requests.post(
        DEEPSTACK_URL+"/v1/vision/face/match", files={"image1": image_data1, "image2": image_data3}, data={"api_key": API_KEY}
    )

    response_json = response.json()

    assert response.status_code == 200, "Request failed with error: {}".format(
        response_json["error"])
    assert response_json["success"] == True
    assert response_json["similarity"] < 0.55


def test_list():

    response = requests.post(
        DEEPSTACK_URL+"/v1/vision/face/list", data={"api_key": API_KEY})
    response_json = response.json()

    assert response.status_code == 200, "Request failed with error: {}".format(
        response_json["error"])
    assert response_json["success"] == True
    assert "Adele" in response_json["faces"]


def test_delete():

    time.sleep(6)

    response = requests.post(
        DEEPSTACK_URL+"/v1/vision/face/delete", data={"userid": "Adele"}
    )
    response_json = response.json()

    assert response.status_code == 200, "Request failed with error: {}".format(
        response_json["error"])
    assert response_json["success"] == True


def test_recognize_after_delete():

    time.sleep(10)

    image_data = open(os.path.join(DATA_DIR, "adele2.jpg"), "rb").read()

    response = requests.post(
        DEEPSTACK_URL+"/v1/vision/face/recognize", files={"image": image_data}, data={"api_key": API_KEY}
    )
    response_json = response.json()

    assert response.status_code == 200, "Request failed with error: {}".format(
        response_json["error"])
    assert response_json["success"] == True
    assert response_json["predictions"][0]["userid"] == "unknown"


def test_list_after_delete():

    time.sleep(10)

    response = requests.post(
        DEEPSTACK_URL+"/v1/vision/face/list", data={"api_key": API_KEY})
    response_json = response.json()

    assert response.status_code == 200, "Request failed with error: {}".format(
        response_json["error"])
    assert response_json["success"] == True
    assert "Adele" not in response_json["faces"]
