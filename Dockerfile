# Use NVIDIA's CUDA base image
FROM python:3.11.12-slim-bullseye

# COPY permissions.sh /usr/local/bin/
# RUN chmod +x /usr/local/bin/permissions.sh
# ENTRYPOINT ["/usr/local/bin/permissions.sh"]

RUN useradd -m -u 1000 lelalomos

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

USER lelalomos


