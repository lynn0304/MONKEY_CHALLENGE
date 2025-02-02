# FROM nvidia/cuda:11.1.1-cudnn8-runtime-ubuntu20.04
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu20.04
ENV DBUS_SESSION_BUS_ADDRESS=/dev/null

ENV TZ=Europe/Amsterdam
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1

RUN : \
&& apt-get update \
&& apt-get install -y git \
&& apt-get install -y --no-install-recommends software-properties-common \
&& add-apt-repository -y ppa:deadsnakes/ppa \
&& apt-get update \
&& apt-get install -y --no-install-recommends python3.10 python3.10-venv python3.10-dev \
&& apt-get clean \
&& :

# Add env to PATH
RUN python3.10 -m venv /venv
ENV PATH=/venv/bin:$PATH

RUN : \
    && apt-get update \
    && apt-get install -y openslide-tools libopenslide0 \
    && apt-get install -y build-essential libffi-dev libxml2-dev libjpeg-turbo8-dev zlib1g-dev \
    && apt-get clean \
    && :

RUN /venv/bin/python3.10 -m pip install --no-cache-dir openslide-python

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
RUN chown -R user:user /venv/
USER user

WORKDIR /opt/ml/model
# COPY --chown=user:user /opt/ml/model /opt/ml/model/

WORKDIR /opt/app
COPY --chown=user:user requirements.txt /opt/app/

RUN /venv/bin/python3.10 -m pip install --upgrade pip wheel
# RUN /venv/bin/python3.10 -m pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
RUN /venv/bin/python3.10 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN /venv/bin/python3.10 -m pip install -U ultralytics sahi

# Install Whole Slide Data
RUN /venv/bin/python3.10 -m pip install 'git+https://github.com/DIAGNijmegen/pathology-whole-slide-data@main'
RUN /venv/bin/python3.10 -m pip install \
    --no-cache-dir \
    -r /opt/app/requirements.txt

# Verify torch installation to ensure it's available
RUN /venv/bin/python3.10 -c "import torch; print(torch.__version__)"
RUN /venv/bin/python3.10 -c "import torch; print(torch.cuda.is_available())"

WORKDIR /opt/app
COPY --chown=user:user inference.py /opt/app/
USER user
ENTRYPOINT ["/venv/bin/python3.10", "inference.py"]