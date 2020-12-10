import requests

image_data = open("test-image5.jpg", "rb").read()

response = requests.post(
    "http://localhost:80/v1/vision/scene", files={"image": image_data}
).json()
print("Label:", response["label"])
print(response)
