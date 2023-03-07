FROM continuumio/miniconda3

WORKDIR /recon

RUN conda install -c conda-forge cupy
RUN conda install -c conda-forge cudnn
RUN conda install -c conda-forge nccl
RUN conda install -c conda-forge numba
RUN conda install -c conda-forge pywavelets
RUN conda install -c conda-forge tqdm
RUN conda install -c conda-forge scipy
RUN conda install -c anaconda pillow
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
