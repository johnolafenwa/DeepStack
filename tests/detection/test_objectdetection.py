from io import open
import os 
import requests

IMAGES_DIR = os.getenv("TEST_IMAGES_DIR")
DEEPSTACK_URL = os.getenv("TEST_DEEPSTACK_URL")
API_KEY = os.getenv("TEST_API_KEY")

def test_detection():

    image_data = open(os.path.join(IMAGES_DIR,"detection.jpg"), "rb").read()

    response = requests.post(
        DEEPSTACK_URL+"/v1/vision/detection",
        files={"image": image_data},data={"api_key": API_KEY}
    ).json()

    assert response["success"] == True
    assert len(response["predictions"]) == 3
