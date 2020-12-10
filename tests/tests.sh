#!/bin/bash
image=$1
echo $image
runtime_args=""
if [[ $image == *"gpu"* ]]
then
runtime_args="--rm --gpus all"
fi
sudo nohup docker run $runtime_args --name=test1 -v localstorage:/datastore -e VISION-FACE=True -p 80:5000 $image & 
sleep 10s
cd tests/face && pytest
sudo nohup docker run $runtime_args --name=test3 -v localstorage:/datastore -e VISION-DETECTION=True -p 80:5000 $image & 
sleep 15s
cd ../detection && pytest
sudo docker container stop test3
sudo nohup docker run $runtime_args --name=test4 -v localstorage:/datastore -e VISION-SCENE=True -p 80:5000 $image & 
sleep 15s
cd ../scene && pytest
sudo docker container stop test4
sudo docker system prune