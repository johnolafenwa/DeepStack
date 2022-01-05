from locust import HttpUser, between, task
from locust.user import wait_time 
import os

DATA_DIR = os.getenv("TEST_DATA_DIR")
DEEPSTACK_URL = os.getenv("TEST_DEEPSTACK_URL")
API_KEY = os.getenv("TEST_API_KEY")


class SceneCaller(HttpUser):

    @task
    def scene(self):

        image_data = open(os.path.join(DATA_DIR,"scene.jpg"), "rb").read()

        self.client.post("/v1/vision/scene",
        files={"image": image_data},data={"api_key": API_KEY})

class ObjectDetectionCaller(HttpUser):

    @task
    def detect(self):

        image_data = open(os.path.join(DATA_DIR,"detection.jpg"), "rb").read()

        self.client.post("/v1/vision/detection",
        files={"image": image_data},data={"api_key": API_KEY})
        