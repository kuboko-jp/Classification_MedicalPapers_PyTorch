FROM nvidia/cuda:11.2.1-devel-ubuntu20.04

RUN apt-get update && apt-get install -y \
    sudo \
    wget \
    vim \
    git
WORKDIR /opt

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    sh Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda3 && \
    rm -r Miniconda3-latest-Linux-x86_64.sh

ENV PATH /opt/miniconda3/bin:$PATH

RUN pip install --upgrade pip && \
    conda update -n base -c defaults conda && \
    conda create -n srws && \
    conda init
ENV CONDA_DEFAULT_ENV srws
RUN echo "conda activate srws" >> ~/.bashrc
ENV PATH /opt/conda/envs/srws/bin:$PATH

#SHELL ["conda", "run", "-n", "srws", "/bin/bash", "-c"]
RUN conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia -y

WORKDIR /

CMD ["/bin/bash"]