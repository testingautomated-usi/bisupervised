# Installation Instruction (aka. "How to run build and run the docker file")

## Step 1: Getting the Docker Image
We provide a pre-built docker image through the github container repository (ghcr.io), allowing a very easy way to get and run the image.
To adhere to the FAIR principles (github may become unavailable one day), or in case you want to make changes to the sources, you can also build the docker image from scratch. In the following, these two options are explained.

### Option 1: Downloading the image from ghcr.io
The easiest way to get the docker image is by downloading it from ghcr.io:

```
docker pull ghcr.io/testingautomated-usi/bisupervised:latest
```


### Option 2: Building the dockerfile from source

You can build the docker image from source by navigating into the cloned replication package and running the following command:

```
docker build . -t ghcr.io/testingautomated-usi/bisupervised
```

Note that building the image will take a couple of minutes and requires an active internet connection.



## Step 2: Running the Docker file in interactive mode

Start the container with the following command (replacing `/path/to/assets/` with the path to the assets folder):

```
docker run -it --rm -v /path/to/bisupervised/generated/:/generated ghcr.io/testingautomated-usi/bisupervised
```
> :information_source: The `-v /path/to/bisupervised/generated/:/generated` part of the command mounts the folder at your local (host) machine path `/path/to/bisupervised/generated/` to the folder `/generated` within the docker container, which allows the scripts to read its contents without copying them explicitely into the docker container, and to write to `/generated` without losing the outputs when destroying the container after its use.

You should now see a Tensorflow welcome message.

Verify that the folder containing the generated artifacts is correctly mounted by running `ls /generated`, anticipating the results as shown in the following:

```bash
root@98d069add304:/src# ls /generated
gpt3  predictions_and_uncertainties  results  trained_models

```

Congrats, you are now ready to start reproducing our experiments, as described in the [README](README.md).

***A note on running with GPU (supported, but not recommended)***

On linux with an nvidia-gpu, you can optionally install the [nvidia-docker toolkit](https://github.com/NVIDIA/nvidia-docker)  which will allow you to use a GPU for training and inference. With the toolkit installed, simply add add `--gpus all` after the `--rm` flag when running the container.
  
Note that for our experiments, a GPU is not really required: We provide all predictions on which our experiments depend in our replication package and model training or invocation is not needed. GPUs are of a cause of errors and we thus recommend running the replication package without GPU.


