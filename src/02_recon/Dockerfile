FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

WORKDIR /recon

RUN conda update -n base -c defaults conda
RUN conda install -c conda-forge numpy=1.21.5
RUN conda install -c conda-forge cupy=11.1.0 cudatoolkit=11.3
RUN conda install -c conda-forge cudnn=8.4.1.50
RUN conda install -c conda-forge nccl=2.14.3.1
RUN conda install -c conda-forge numba=0.55.2
RUN conda install -c conda-forge pywavelets=1.3.0
RUN conda install -c conda-forge tqdm=4.64.1
RUN conda install -c conda-forge scipy=1.7.3
RUN conda install -c conda-forge sympy=1.10.1
RUN conda install -c conda-forge git=2.37.3
RUN pip install git+https://github.com/mikgroup/sigpy.git@master

COPY load_data.py ./
COPY mrf.py ./
COPY prox.py ./
COPY optalg.py ./
COPY optpoly.py ./
COPY main.py ./

ENTRYPOINT ["python3", "main.py"]
