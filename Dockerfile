# Use NVIDIA's CUDA base image
FROM python:3.11.12-slim-bullseye

USER root

ENV TZ="Asia/Bangkok"
RUN apt-get -y update
RUN apt-get -y upgrade
RUN apt-get -y install gcc build-essential wget

COPY install_talib.sh /app/

RUN chmod 777 /app/install_talib.sh
RUN /app/install_talib.sh

# Install required Python packages
COPY requirements.txt /app/
RUN pip install --default-timeout=100 -r /app/requirements.txt
RUN pip install --upgrade pip

# Copy application code
COPY . /app

WORKDIR /app


