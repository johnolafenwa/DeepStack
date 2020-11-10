import requests
from PIL import Image

image_data = open("test-image3.jpg","rb").read()
image = Image.open("test-image3.jpg").convert("RGB")

response = requests.post("http://localhost:80/v1/vision/detection",files={"image":image_data}).json()
print(response)
i = 0
for object in response["predictions"]:

    label = object["label"]
    y_max = int(object["y_max"])
    y_min = int(object["y_min"])
    x_max = int(object["x_max"])
    x_min = int(object["x_min"])
    cropped = image.crop((x_min,y_min,x_max,y_max))
    cropped.save("image{}_{}.jpg".format(i,label))

    i += 1