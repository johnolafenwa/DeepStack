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
cd face && pytest
sudo docker container rm -f test1
sudo nohup docker run $runtime_args --name=test2 -v localstorage:/datastore -e VISION-DETECTION=True -p 80:5000 $image & 
sleep 15s
cd ../detection && pytest
sudo docker container rm -f test2
sudo nohup docker run $runtime_args --name=test3 -v localstorage:/datastore -e VISION-SCENE=True -p 80:5000 $image & 
sleep 15s
cd ../scene && pytest
sudo docker container rm -f test3