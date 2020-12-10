import requests
from io import  open



def test_onnx():
    model = open("idenprof.onnx","rb").read()
    config = open("config-onnx.json","rb").read()

    response = requests.post("http://localhost:80/v1/vision/addmodel",files={"model":model,"config":config},data={"name":"professiontestonnx","api_key":"Mojohn1"}).json()
    assert response["success"] == True


