# DeepStack
The World's Leading Cross Platform AI Engine for Edge Devices, with over `3.2 million` installs on **Docker Hub**.

[![black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](hhttps://github.com/johnolafenwa/DeepStack/blob/dev/LICENSE)  

**Website**: [https://deepstack.cc](https://deepstack.cc)

**Documentation**: [https://docs.deepstack.cc](https://docs.deepstack.cc)

**Forum**: [https://forum.deepstack.cc](https://forum.deepstack.cc)

**Dev Center**: [https://dev.deepstack.cc](https://dev.deepstack.cc/)

**DeepStack** is owned and maintained by [DeepQuest AI](https://www.deepquestai.com/).


# Introduction
DeepStack is an AI API engine that serves pre-built models and custom models on multiple edge devices locally or on your private cloud. Supported platforms are:

- **Linux OS** via Docker ( CPU and NVIDIA GPU support )
- **Mac OS** via Docker
- **Windows 10** ( native application )
- **NVIDIA Jetson** via Docker.

DeepStack runs completely offline and independent of the cloud. You can also install and run DeepStack on any cloud VM with docker installed to serve as your private, state-of-the-art and real-time AI server.

# Installation and Usage
Visit [https://docs.deepstack.cc/getting-started](https://docs.deepstack.cc/getting-started/) for installation instructions. The documentation provides example codes for the following programming languages with more to be added soon.

- **Python**
- **C#**
- **NodeJS**

# Build from Source (For Docker Version)

- **Install Prerequisites**

    - [Install Golang](https://golang.org)
    - [Install Docker](https://docker.com)
    - [Install GIT](https://git-scm.com)
    - [Install GIT LFS](https://github.com/git-lfs/git-lfs/wiki/Installation)
    - [Install Redis Server](https://redis.io/)
    - [Install Python3.7](https://python.org)
    - [Install Powershell 7+](https://docs.microsoft.com/en-us/powershell/scripting/windows-powershell/install/installing-windows-powershell?view=powershell-7.1)

- **Clone DeepStack Repo** 

    ```git clone https://github.com/johnolafenwa/DeepStack.git```

- **CD to DeepStack Repo Dir**

    ```cd DeepStack```

- **Fetch Repo Files**

    ``git lfs pull``

- **Build DeepStack Server**

    ```cd server && go build```

- **Build DeepStack CPU Version**

    ```cd .. && sudo docker build -t deepquestai/deepstack:cpu . -f Dockerfile.cpu```

- **Build DeepStack GPU Version**

    ```sudo docker build -t deepquestai/deepstack:gpu . -f Dockerfile.gpu```

- **Build DeepStack Jetson Version**

    ```sudo docker build -t deepquestai/deepstack:jetpack . -f Dockerfile.gpu-jetpack```

- **Running and Testing Locally Without Building**
    - Unless you wish to install requirements system wide, create a virtual environment with ```python3.7 -m venv venv``` and activate with ```source venv/bin/activate```

    - Install Requirements with ```pip3 install -r requirements.txt```

    - For CPU Version, Install PyTorch with ```pip3 install torch==1.6.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html```

    - For GPU Version, Install Pytorch with ```pip3 install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html```

    - Start Powershell
        ```pwsh```

    - For CPU Version, Run ```.\setup_docker_cpu.ps1```

    - For GPU Version, Run ```.\setup_docker_gpu.ps1```

    - CD To Server Dir
        ```cd server```

    - Build DeepStack Server
        ```go build```

    - Set Any of the APIS to enable;
        ```$env:VISION_DETECTION = "True"```, ```$env:VISION_FACE = "True"```, ```$env:VISION_SCENE = "True"```

    - Run DeepStack
        ```.\server```

    You can find all logs in the ```directory``` in the repo root.
    Note that DeepStack will be running on the default port ```5000```.

# Integrations and Community
The DeepStack ecosystem includes a number of popular integrations and libraries built to expand the functionalities of the AI engine to serve IoT, industrial, monitoring and research applications. A number of them are listed below

- [HASS-DeepStack-Object](https://github.com/robmarkcole/HASS-Deepstack-object)
- [HASS-DeepStack-Face](https://github.com/robmarkcole/HASS-Deepstack-face)
- [HASS-DeepStack-Scene](https://github.com/robmarkcole/HASS-Deepstack-scene)
- [DeepStack with Blue Iris - YouTube video](https://www.youtube.com/watch?v=fwoonl5JKgo)
- [DeepStack with Blue Iris - Forum Discussion](https://ipcamtalk.com/threads/tool-tutorial-free-ai-person-detection-for-blue-iris.37330/)
- [DeepStack on Home Assistant](https://community.home-assistant.io/t/face-and-person-detection-with-deepstack-local-and-free/92041)
- [DeepStack-UI](https://github.com/robmarkcole/deepstack-ui)
- [DeepStack-Python Helper](https://github.com/robmarkcole/deepstack-python)
- [DeepStack-Analytics](https://github.com/robmarkcole/deepstack-analytics)
- [DeepStackAI Trigger](https://github.com/danecreekphotography/node-deepstackai-trigger)


# Contributors Guide
(coming soon)