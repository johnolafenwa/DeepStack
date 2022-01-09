from io import open, BytesIO
from PIL import Image
import os 
import requests
import base64

DATA_DIR = os.getenv("TEST_DATA_DIR")
DEEPSTACK_URL = os.getenv("TEST_DEEPSTACK_URL")
API_KEY = os.getenv("TEST_API_KEY")

def test_enhance():

    image_data = open(os.path.join(DATA_DIR,"fox_small.jpg"), "rb").read()

    response = requests.post(
        DEEPSTACK_URL+"/v1/vision/enhance",
        files={"image": image_data},data={"api_key": API_KEY}
    )
    response_json = response.json()

    assert response.status_code == 200, "Request failed with error: {}".format(response_json["error"])

    assert response_json["success"] == True
    assert type(response_json["width"]) == int
    assert type(response_json["height"]) == int
    assert len(response_json["base64"]) > 1000

    image4X_byte = base64.b64decode(response_json["base64"])
    image = Image.open(BytesIO(image4X_byte))
    width, height = image.size

    assert width == 1200
    assert height == 632
