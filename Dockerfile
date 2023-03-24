FROM tensorflow/tensorflow:2.9.1-gpu

# Install dependencies
RUN /usr/bin/python3 -m pip install pip==23.0.1
COPY requirements.txt ./requirements.txt
RUN pip install -r ./requirements.txt
RUN rm ./requirements.txt

# Required to resolve cudnn issues
RUN apt install --allow-change-held-packages libcudnn8=8.1.0.77-1+cuda11.2 -y

# Copy replication package sources
WORKDIR /
COPY src src
ENV PYTHONPATH "${PYTHONPATH}:$PWD"
