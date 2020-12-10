import requests
from PIL import Image,ImageDraw
import os


files = os.listdir(".")

for f in files:

    if f.endswith(".jpg") or f.endswith(".png"):

        image_data = open(f,"rb").read()
        image = Image.open(f).convert("RGB")

        response = requests.post("http://localhost:80/v1/vision/detection",files={"image":image_data}, data={"min_confidence":0.7}).json()
        print(response)
        draw = ImageDraw.Draw(image)
        for object in response["predictions"]:

            label = object["label"]
            conf = object["confidence"]
            y_max = int(object["y_max"])
            y_min = int(object["y_min"])
            x_max = int(object["x_max"])
            x_min = int(object["x_min"])                                           
            
            draw.rectangle([(x_min,y_min),(x_max,y_max)],outline="red",width=5)
            draw.text((x_min,y_min),"{}".format(label))
            draw.text((x_min,y_min-10),"{}".format(conf))
            image.save("detected.jpg")
