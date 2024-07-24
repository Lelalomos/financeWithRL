FROM nvidia/cuda:11.4.3-cudnn8-devel-ubuntu20.04
ENV TZ="Asia/Bangkok"
RUN echo 'DEBIAN_FRONTEND=noninteractive' >> /etc/environment
ENV DEBIAN_FRONTEND=noninteractive
RUN apt -y update
RUN apt -y upgrade
RUN apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
RUN apt-get -y clean
COPY . /app
WORKDIR /app
RUN pip3 install -r requirements.txt


