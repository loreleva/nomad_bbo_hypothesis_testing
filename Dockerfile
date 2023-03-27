FROM python:latest
COPY ./requirements.txt .

RUN apt update -y
RUN apt install -y r-base cmake mpich
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY ./nomad-v.4.3.1 /nomad-v.4.3.1
WORKDIR /nomad-v.4.3.1
RUN export NOMAD_HOME=/nomad-v.4.3.1
RUN cmake -DBUILD_INTERFACE_PYTHON=ON -DTEST_OPENMP=OFF -S . -B build/release
RUN cmake --build build/release --parallel 4
RUN cmake --install build/release