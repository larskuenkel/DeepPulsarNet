#Currently Dockerfile does not do a whole lot since the requirements changed

# PyTorch Base Image
FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime

# metainformation
LABEL version="0.1"

# avoid stuck build due to user prompt
ARG DEBIAN_FRONTEND=noninteractive

# Essential Installs
RUN apt-get update
RUN apt-get install --no-install-recommends -y \
    build-essential \
    gfortran \
    autoconf \
    libomp-dev \
    libcfitsio-dev \
    libltdl-dev \
    python3-dev \
    python3-numpy \
    python3-pip \
    automake \
    libtool \
    unzip \
    make \
    cmake \
    fftw3 \
    fftw3-dev \
    tcsh \
    curl \
    gcc \
    git \
    wget \
    latex2html \
    libcfitsio-bin \
    libfftw3-bin \
    libfftw3-dev \
    libglib2.0-dev \
    libpng-dev \
    libx11-dev \
    pgplot5 \
    tcsh \
    unzip \
    sudo \
    vim \
    && rm -rf /var/lib/apt/lists/*
RUN apt-get clean

# Add pgplot environment variables
ENV PGPLOT_DIR=/usr/local/pgplot
ENV PGPLOT_DEV=/Xserve

# Install python dependancies
RUN pip3 install numpy \
    scipy \
    astropy

COPY . .
RUN pip3 install --no-cache-dir --upgrade wheel pip
RUN pip3 install --no-cache-dir -r requirements.txt

ADD . /home/DeepPulsarNet

RUN mkdir -p /home/soft
ENV HOME=/home
ENV PSRHOME=/home/soft
ENV OSTYPE=linux

#WORKDIR /home/
#RUN mkdir data
#RUN wget https://github.com/larskuenkel/SKA-TestVectorGenerationPipeline/raw/master/ASC/ASC.zip
#RUN unzip -n -q ASC.zip \ 
#    -d data/all_asc/ && \
#    rm ASC.zip
# Installs all the C dependancies -----------------------------
WORKDIR /home/soft

#copy respositories

#currently no presto installation since tests fail
#RUN git clone https://github.com/scottransom/presto.git

WORKDIR /home/soft


## Install presto python scripts
#ENV PRESTO /home/soft/presto
#ENV LD_LIBRARY_PATH /home/soft/presto/lib
#ADD . /home/soft/presto
#WORKDIR /home/soft/presto/src
## The following is necessary if your system isn't Ubuntu 20.04
#RUN make cleaner
## Now build from scratch
#RUN make libpresto slalib
#WORKDIR /home/soft/presto
#    RUN pip3 install /home/soft/presto
#    RUN sed -i 's/env python/env python3/' /home/soft/presto/bin/*py
#    RUN python3 tests/test_presto_python.py 
##python tests for presto fail currently, skip presto install for now

# Install psrcat
RUN wget https://www.atnf.csiro.au/research/pulsar/psrcat/downloads/psrcat_pkg.tar.gz && \
    gunzip psrcat_pkg.tar.gz && \
    tar -xvf psrcat_pkg.tar && \
    rm psrcat_pkg.tar && \
    cd psrcat_tar && \
    ls && \
    bash makeit && \
    cp psrcat /usr/bin
ENV PSRCAT_FILE /home/soft/psrcat_tar/psrcat.db
    
# Install tempo
#RUN git clone https://github.com/nanograv/tempo.git && \
#    cd tempo && \
#    ./prepare && \
#    ./configure && \
#    make && \
#    make install
#ENV TEMPO /home/soft/tempo

# Install tempo2
#RUN sudo git clone https://bitbucket.org/psrsoft/tempo2.git && cd tempo2 && \
#    sudo ./bootstrap && \
#    sudo cp -r T2runtime ${PSRHOME}/tempo2/share/
#ENV TEMPO2="${PSRHOME}/tempo2/share/"
#RUN cd ${PSRHOME}/tempo2 && sudo TEMPO2=${TEMPO2} ./configure --prefix=/usr/local && sudo make && sudo make install && \
#    sudo make plugins && sudo make plugins-install && sudo make clean && sudo make distclean
#ENV LD_LIBRARY_PATH=/usr/local/lib/:${LD_LIBRARY_PATH}


WORKDIR /home/
