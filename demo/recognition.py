import requests

test_image = open("test-image.jpg","rb").read()

res = requests.post("http://localhost:80/v1/vision/face/recognize",
files={"image":test_image}, data={"min_confidence":0.1}).json()
print(res)

for user in res["predictions"]:
    print(user["userid"])
