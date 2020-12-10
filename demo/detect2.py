import requests
from PIL import Image,ImageDraw

image_data = open("detected.jpg","rb").read()
#image = Image.open("/home/johnolafenwa/Downloads/1_zoS8E7Gooe_-3Hi43_R_Yg.jpeg").convert("RGB")

response = requests.post("http://localhost:80/v1/vision/detection",files={"image":image_data}).json()
print(response)
i = 0
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


