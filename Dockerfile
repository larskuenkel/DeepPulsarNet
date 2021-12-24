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
    sudo \
    && rm -rf /var/lib/apt/lists/*
RUN apt-get clean

# Install pip requirements
COPY . .
RUN pip3 install --no-cache-dir --update wheel pip
RUN pip install --no-cache-dir -r requirements.txt

# Define home, psrhome, OSTYPE
RUN mkdir -p /home/psr/soft
ENV HOME=/home
ENV PSRHOME=/home/psr/soft
ENV OSTYPE=linux

# psrcat
ENV PSRCAT_FILE=$PSRHOME/psrcat_tar/psrcat.db
ENV PATH=$PATH:$PSRHOME/psrcat_tar

# Tempo
ENV TEMPO=$PSRHOME/tempo
ENV PATH=$PATH:$PSRHOME/tempo/bin

# Tempo2
ENV TEMPO2=$PSRHOME/tempo2/T2runtime
ENV PATH=$PATH:$PSRHOME/tempo2/T2runtime/bin
ENV C_INCLUDE_PATH=$C_INCLUDE_PATH:$PSRHOME/tempo2/T2runtime/include
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PSRHOME/tempo2/T2runtime/lib

# SIGPROC
# These flags assist with the Sigproc compilation process, so do not remove them. If you take
# them out, then Sigproc will not build correctly.
ENV SIGPROC=$PSRHOME/sigproc
ENV PATH=$PATH:$SIGPROC/install/bin
ENV FC=gfortran
ENV F77=gfortran
ENV CC=gcc
ENV CXX=g++

##############################################################################################
# TEST VECTOR PIPELINE + README Setup
##############################################################################################
# Download test vector files
WORKDIR /home
RUN wget https://github.com/scienceguyrob/SKA-TestVectorGenerationPipeline/raw/master/Dockerfile/CentOS/DockerImageReadme.txt && \
    cd $PSRHOME && \
    wget https://github.com/scienceguyrob/SKA-TestVectorGenerationPipeline/raw/master/Deploy/pulsar_injection_pipeline.zip && \
    unzip pulsar_injection_pipeline.zip -d $PSRHOME && \
    rm __MACOSX -R && \
    rm pulsar_injection_pipeline.zip

##############################################################################################
# Elmarie van Heerden's Code
##############################################################################################

RUN mkdir /home/psr/soft/evh && \
    cd $PSRHOME/evh && \
    wget https://raw.githubusercontent.com/EllieVanH/PulsarDetectionLibrary/master/Readme_For_Ersartz.txt && \
    wget https://raw.githubusercontent.com/EllieVanH/PulsarDetectionLibrary/master/ersatz.py

##############################################################################################
# PULSAR SOFTWARE PIPELINE
##############################################################################################
WORKDIR /home

# Download the software
RUN wget https://github.com/scienceguyrob/SKA-TestVectorGenerationPipeline/raw/master/Deploy/Software/08_12_2016/Sigproc_MJK_SNAPSHOT_08_12_2016.zip && \
    wget https://github.com/scienceguyrob/SKA-TestVectorGenerationPipeline/raw/master/Deploy/Software/08_12_2016/Tempo_SNAPSHOT_08_12_2016.zip && \
    wget https://github.com/scienceguyrob/SKA-TestVectorGenerationPipeline/raw/master/Deploy/Software/08_12_2016/Tempo2_2016.11.3_SNAPSHOT_08_12_2016.zip && \
# Unzip the software
    unzip Sigproc_MJK_SNAPSHOT_08_12_2016.zip -d /home/Sigproc_MJK_SNAPSHOT_08_12_2016 && \
    unzip Tempo_SNAPSHOT_08_12_2016.zip -d /home/Tempo_SNAPSHOT_08_12_2016 && \
    unzip Tempo2_2016.11.3_SNAPSHOT_08_12_2016.zip -d /home/Tempo2_2016.11.3_SNAPSHOT_08_12_2016 && \
# Remove zip files
    rm Sigproc_MJK_SNAPSHOT_08_12_2016.zip && \  
    rm Tempo_SNAPSHOT_08_12_2016.zip && \
    rm Tempo2_2016.11.3_SNAPSHOT_08_12_2016.zip && \
# Move the software to the correct folder location
    mv /home/Sigproc_MJK_SNAPSHOT_08_12_2016 /home/psr/soft/sigproc && \
    mv /home/Tempo_SNAPSHOT_08_12_2016 /home/psr/soft/tempo && \
    mv /home/Tempo2_2016.11.3_SNAPSHOT_08_12_2016 /home/psr/soft/tempo2 && \
# Remove files which appear due to me packaging on a MAC.
    rm /home/psr/soft/sigproc/__MACOSX -R && \
    rm /home/psr/soft/tempo/__MACOSX -R && \
    rm /home/psr/soft/tempo2/__MACOSX -R

##############################################################################################
# TEMPO Installation
##############################################################################################
WORKDIR $PSRHOME/tempo
RUN ./prepare && \
    ./configure --prefix=$PSRHOME/tempo && \
    make && \
    make install && \
    mv obsys.dat obsys.dat_ORIGINAL && \
    wget https://raw.githubusercontent.com/mserylak/pulsar_docker/2f15b0d01b922d882b67ec32674d162f41b80377/tempo/obsys.dat

##############################################################################################
# TEMPO2 Installation
##############################################################################################
# Ok here we install the latest version of TEMPO2.

WORKDIR $PSRHOME/tempo2
RUN ./bootstrap && \
    ./configure --x-libraries=/usr/lib/x86_64-linux-gnu --enable-shared --enable-static --with-pic F77=gfortran && \
    make && \
    make install && \
    make plugins-install
WORKDIR $PSRHOME/tempo2/T2runtime/observatory
RUN mv observatories.dat observatories.dat_ORIGINAL && \
    mv oldcodes.dat oldcodes.dat_ORIGINAL && \
    mv aliases aliases_ORIGINAL && \
    wget https://raw.githubusercontent.com/mserylak/pulsar_docker/2f15b0d01b922d882b67ec32674d162f41b80377/tempo2/observatories.dat && \
    wget https://raw.githubusercontent.com/mserylak/pulsar_docker/2f15b0d01b922d882b67ec32674d162f41b80377/tempo2/aliases

##############################################################################################
# Sigproc Installation
##############################################################################################
# Ok here we install sigproc - This is Mike Keith's version of Sigproc, which comes with the
# fast_fake utility. First we set the environment variables for the install, then execute the
# building steps.
WORKDIR $SIGPROC
RUN ./bootstrap && \
    ./configure --prefix=$SIGPROC/install LDFLAGS="-L/home/psr/soft/tempo2/T2runtime/lib" LIBS="-ltempo2" && \
    make && \
    make install
##############################################################################################
# Finally...
##############################################################################################
# Define the command that will be exectuted when docker runs the container.
WORKDIR /home
ENTRYPOINT /bin/bash