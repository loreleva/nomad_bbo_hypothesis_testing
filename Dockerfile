FROM python:latest
COPY ./requirements.txt .

#DEBUG LINE
COPY ./nomad-v.4.3.1 ./nomad-v.4.3.1
COPY ./code ./code

RUN apt update -y
RUN apt install -y r-base cmake mpich
RUN pip install --upgrade pip
RUN pip install -r requirements.txt