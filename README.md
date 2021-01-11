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

# Build from Source

- **Install Prerequisites**

    - [Install Golang](https://golang.org)
    - [Install Docker](https://docker.com)
    - [Install GIT](https://git-scm.com)
    - [Install GIT LFS](https://github.com/git-lfs/git-lfs/wiki/Installation)

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


# Contributors Guide
(coming soon)