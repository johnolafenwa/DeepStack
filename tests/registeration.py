import requests

def register_face(img_path,user_id):
    image_data = open(img_path,"rb").read()
    response = requests.post("http://localhost:5000/v1/vision/face/register",
    files={"image":image_data}, data={"userid":user_id}).json()
    print(response)

register_face("cruise.jpg","Tom Cruise")
register_face("adele.jpg","Adele")
register_face("elba.jpg","Idris Elba")
register_face("perri.jpg","Christina Perri")