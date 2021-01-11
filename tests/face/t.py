import requests


def test_register():

    image_data = open("adele1.jpg", "rb").read()

    response = requests.post(
        "http://localhost:80/v1/vision/face",
        files={"image": image_data},
        data={"userid": "Adele"},
    ).json()

    print(response)


test_register()
