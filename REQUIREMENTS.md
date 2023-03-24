# Requirements 

## Software Requirements

There's only two things you need on your machine to get started:
- We provide a dockerfile and a docker image for our reproduction package. Thus, you do not need to install any specific dependencies other than docker.
- You need to clone (`git clone https://github.com/testingautomated-usi/bisupervised.git`) our replication package.
(Adhereing to the FAIR principles, our github repository is archived/mirrored on zenodo.org and you can also always download it from there).
In the following, we will refer to '/path/to/bisupervised/' as the folder containing the replication package.

Note: If you're curious about the dependencies included in the docker image, see our [dockerfile](Dockerfile) and [requirments.txt](requirements.txt).

## Hardware Requirements
We tested our replication package it on two machines, a workstation (64GB RAM, 12 Cores, RTX3090 GPU)
and on a notebook (32GB RAM, 4 Cores). Both machines are using Ubuntu 22.04.

*Other hardware and operating systems:* We do assume that our replication package also works on machines with lower specs (e.g. machines with only 16GB of RAM)
and other operating systems (thanks to dockerization), but we were not able to test this.

*Model training and prediction collection:* We provide all predictions on which our experiments depend in our replication package and model training or invocation is not needed. Model training and inference is also not part of our contributions and assumed as given. Still, for completeness, we included scripts to (dependent on the case study) download or train models, make inferences or invoke the openai API to make remote predictions. If you plan on conducting model-retraining or inference, a strong machine (as our workstation) might be desired or needed.
