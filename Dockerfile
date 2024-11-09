# Use NVIDIA's CUDA base image
FROM python:3.13.0-slim-bullseye

ENV TZ "Asia/Bangkok"
RUN apt-get -y update
RUN apt-get -y upgrade
RUN apt-get -y install gcc build-essential wget
RUN pip install --upgrade pip

# Install required Python packages
COPY requirements.txt /app/
RUN pip install --default-timeout=100 -r /app/requirements.txt

# Copy application code
COPY . /app

WORKDIR /app


