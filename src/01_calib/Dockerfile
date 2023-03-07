FROM continuumio/miniconda3:4.12.0

WORKDIR /recon

RUN conda install -c conda-forge cupy=8.3.0
RUN conda install -c conda-forge cudnn=7.6.5.32
RUN conda install -c conda-forge nccl=2.8.3.1
RUN conda install -c conda-forge numba=0.55.1
RUN conda install -c conda-forge pywavelets=1.3.0 
RUN conda install -c conda-forge tqdm=4.64.1
RUN conda install -c conda-forge scipy=1.8.1
RUN conda install -c anaconda pillow=9.0.1
RUN pip install git+https://github.com/mikgroup/sigpy.git@master
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install --upgrade setuptools pip
RUN pip install opencv-python
RUN apt-get update
RUN apt-get -y install make gcc libfftw3-dev liblapacke-dev libpng-dev libopenblas-dev \
    && wget https://github.com/mrirecon/bart/archive/v0.8.00.tar.gz \
    && tar xzvf v0.8.00.tar.gz \
    && cd bart-0.8.00 \
    && make

COPY rovir.py ./
COPY autofov.py ./
COPY cfl.py ./
COPY main.py ./

ENTRYPOINT ["python3", "main.py"]
