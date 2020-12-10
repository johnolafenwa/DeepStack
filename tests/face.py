import requests
from PIL import Image

image_data = open(
    "/home/johnolafenwa/Documents/source/jetson/DeepStack/tests/test-image6.jpeg", "rb"
).read()
image = Image.open(
    "/home/johnolafenwa/Documents/source/jetson/DeepStack/tests/test-image6.jpeg"
).convert("RGB")

response = requests.post(
    "http://localhost:80/v1/vision/face/recognize", files={"image": image_data}
).json()
print(response)
i = 0
for face in response["predictions"]:

    y_max = int(face["y_max"])
    y_min = int(face["y_min"])
    x_max = int(face["x_max"])
    x_min = int(face["x_min"])
    cropped = image.crop((x_min, y_min, x_max, y_max))

    cropped.save("image{}.jpg".format(i))

    i += 1
