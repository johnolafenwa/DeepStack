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

class FaceDetectionCaller(HttpUser):

    @task
    def detect(self):

        image_data = open(os.path.join(DATA_DIR,"face_detection.jpg"), "rb").read()

        self.client.post("/v1/vision/face",
        files={"image": image_data},data={"api_key": API_KEY})

class FaceRecognition(HttpUser):

    @task()
    def recognize(self):

        image_data = open(os.path.join(DATA_DIR,"adele2.jpg"), "rb").read()

        self.client.post("/v1/vision/face/recognize",
        files={"image": image_data},data={"api_key": API_KEY})
        