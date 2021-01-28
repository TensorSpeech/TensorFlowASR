FROM tensorflow/tensorflow:2.3.0-gpu

RUN apt-get update \
&& apt-get upgrade -y \
&& apt-get install -y \
&& apt-get -y install apt-utils gcc libpq-dev libsndfile-dev git build-essential cmake screen

# Clear cache
RUN apt clean && apt-get clean

# Install dependencies
COPY requirements.txt /
RUN pip --no-cache-dir install -r /requirements.txt

# Install rnnt_loss
COPY scripts /scripts
RUN export CUDA_HOME=/usr/local/cuda \
&& ./scripts/install_rnnt_loss.sh

RUN echo "export LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" >> /root/.bashrc