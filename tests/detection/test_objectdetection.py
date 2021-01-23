from io import open
import os 
import requests

DATA_DIR = os.getenv("TEST_DATA_DIR")
DEEPSTACK_URL = os.getenv("TEST_DEEPSTACK_URL")
API_KEY = os.getenv("TEST_API_KEY")

def test_detection():

    image_data = open(os.path.join(DATA_DIR,"detection.jpg"), "rb").read()

    response = requests.post(
        DEEPSTACK_URL+"/v1/vision/detection",
        files={"image": image_data},data={"api_key": API_KEY}
    )
    response_json = response.json()

    assert response.status_code == 200, "Request failed with error: {}".format(response_json["error"])

    assert response_json["success"] == True
    assert len(response_json["predictions"]) == 3
