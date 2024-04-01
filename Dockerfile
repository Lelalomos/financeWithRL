FROM python:3.10.14-slim-bullseye
ENV TZ "Asia/Bangkok"
RUN apt -y update
RUN apt -y upgrade
RUN apt-get -y clean
COPY . /app
WORKDIR /app
RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt