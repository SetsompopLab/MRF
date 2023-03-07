FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

WORKDIR /recon

RUN conda update -n base -c defaults conda
RUN conda install -c conda-forge numpy
RUN conda install -c conda-forge cupy cudatoolkit=11.3
RUN conda install -c conda-forge cudnn
RUN conda install -c conda-forge nccl
RUN conda install -c conda-forge numba
RUN conda install -c conda-forge pywavelets
RUN conda install -c conda-forge tqdm
RUN conda install -c conda-forge scipy
RUN conda install -c conda-forge sympy
RUN conda install -c conda-forge git
RUN pip install git+https://github.com/mikgroup/sigpy.git@master

COPY load_data.py ./
COPY mrf.py ./
COPY prox.py ./
COPY optalg.py ./
COPY optpoly.py ./
COPY main.py ./

ENTRYPOINT ["python3", "main.py"]
