FROM nvidia/cuda:10.0-cudnn7-runtime-ubuntu18.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH=/opt/conda/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    git \
    libglib2.0-0 \
    libgl1-mesa-glx \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh \
    && conda clean -afy

WORKDIR /workspace
COPY environment.yml requirements-demo.txt ./
RUN conda env create -f environment.yml \
    && conda run -n retinanet python -m pip install -r requirements-demo.txt \
    && conda clean -afy

COPY . .

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "retinanet"]
CMD ["python", "predict.py", "--help"]
