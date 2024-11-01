FROM nvidia/cuda:12.2.0-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Tokyo
ENV UV_INSTALL_DIR=/opt

WORKDIR /app

# install basic dependencies
RUN <<EOF
apt-get update
apt-get upgrade -y
apt-get install -y --no-install-recommends software-properties-common
add-apt-repository ppa:deadsnakes/ppa
apt-get install -y --no-install-recommends sudo git curl gcc \
openssh-client python3-distutils
apt-get clean
rm -rf /var/lib/apt/lists/*
EOF

# install uv
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="$UV_INSTALL_DIR/bin:$PATH"
# install python using uv
RUN uv python install 3.12
# cache用にsrc以外を先にinstall
COPY pyproject.toml /app/
RUN uv sync
# srcをコピーしてinstall
COPY src /app/src
RUN uv pip install -e .
COPY . /app/

