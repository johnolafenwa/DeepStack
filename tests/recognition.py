import requests

test_image = open("test-image.jpg","rb").read()

res = requests.post("http://localhost:5000/v1/vision/face/recognize",
files={"image":test_image}).json()

for user in res["predictions"]:
    print(user["userid"])