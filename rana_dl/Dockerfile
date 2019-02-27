FROM tensorflow/tensorflow:latest-gpu-py3
MAINTAINER Avery Uslaner "avery.uslaner@redbutte.utah.edu"

ARG DEBIAN_FRONTEND=noninteractive

ENV PYTHONUNBUFFERED 1

# Some of these packages are already included in the base image
# but are included here for completeness in case that changes in
# future cuda image builds
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libatlas-base-dev \
    libavcodec-dev \
    libavformat-dev \
    libgtk-3-dev \
    libjpeg-dev \
    libpng-dev \
    libswscale-dev \
    libv4l-dev \
    libtiff-dev \
    libx264-dev \
    libxvidcore-dev \
    nano \
    pkg-config \
    python3-dev \
    python3-pip \
    unzip \
    wget \
    zip \
    zlib1g-dev

# Install OpenCV
RUN mkdir /Downloads
WORKDIR /Downloads

RUN wget -O opencv.zip https://github.com/opencv/opencv/archive/3.4.3.zip
RUN wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/3.4.3.zip

RUN unzip opencv.zip
RUN unzip opencv_contrib.zip

WORKDIR /Downloads/opencv-3.4.3
RUN mkdir /Downloads/opencv-3.4.3/build
WORKDIR /Downloads/opencv-3.4.3/build

RUN pip3 install --upgrade pip
RUN pip3 install numpy==1.14.5

RUN cmake \
    -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D WITH_CUDA=OFF \
    -D INSTALL_PYTHON_EXAMPLES=ON \
    -D OPENCV_EXTRA_MODULES_PATH=/Downloads/opencv_contrib-3.4.3/modules \
    -D PYTHON3_EXECUTABLE=/usr/bin/python3 \
    -D PYTHON3_PACKAGES_PATH=/usr/local/lib/python3.5/dist-packages \
    -D BUILD_EXAMPLES=ON ..

RUN make -j7
RUN make install
RUN ldconfig

RUN ln -s /usr/local/lib/python3.5/dist-packages/cv2.cpython-36m-x86_64-linux-gnu.so \
    /usr/local/lib/python3.5/dist-packages/cv2.so

# Copy our code over
RUN mkdir /code
WORKDIR /code

COPY . /code/

RUN pip3 install --no-cache-dir -r /code/requirements.txt

# Add user so that container does not run as root
RUN useradd -m docker
RUN echo "docker:test" | chpasswd
RUN usermod -s /bin/bash docker
RUN usermod -aG sudo docker
RUN chown -R docker:docker /code/

# Create mount points for UofU CHPC Resources
RUN mkdir /uufs /scratch

# Create directories expected by config file
RUN mkdir -p /home/docker/data/output/vggnet
RUN mkdir /home/docker/data/rec
RUN mkdir /home/docker/data/lists

CMD /bin/bash