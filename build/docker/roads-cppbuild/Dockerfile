#FROM ubuntu:18.04
FROM nvidia/cuda:9.2-cudnn7-devel-ubuntu18.04

RUN apt-get update

RUN apt-get install -y \
    libcurl4-openssl-dev \
    liblz4-dev \
    zlib1g-dev \
    libjpeg-turbo8-dev

# Necessary for xorg-dev
RUN apt-get install -y xorg-dev

#RUN cd third_party/glfw \
# && mkdir build \
# && cd build \
# && cmake -DBUILD_SHARED_LIBS=OFF -DGLFW_BUILD_EXAMPLES=OFF -DGLFW_BUILD_TESTS=OFF -DGLFW_BUILD_DOCS=OFF .. \
# && make -j

# I think on 18.04 this is included with the cuda toolkit
#RUN apt-get install -y libnppi7.5

RUN apt-get install -y git

RUN git clone https://git.videolan.org/git/ffmpeg/nv-codec-headers.git /build/nv-codec-headers \
 && cd /build/nv-codec-headers \
 && make \
 && make install

# github mirror is faster than git.ffmpeg.org
#RUN git clone https://git.ffmpeg.org/ffmpeg.git /build/ffmpeg
RUN git clone --depth 1 --branch n4.0.2 https://github.com/FFmpeg/FFmpeg.git /build/fmmpeg
RUN mv /build/fmmpeg /build/ffmpeg

RUN apt-get install -y yasm

RUN cd /build/ffmpeg \
 && ./configure --enable-pic --disable-static --enable-shared --enable-cuda --enable-cuvid --enable-nvenc --extra-cflags=-I/usr/local/cuda/include --extra-ldflags=-L/usr/local/cuda/lib64 \
 && make \
 && make install

RUN git clone --depth 1 --branch 3.4.1 https://github.com/opencv/opencv.git /build/opencv \
 && git clone --depth 1 --branch 3.4.1 https://github.com/opencv/opencv_contrib.git /build/opencv_contrib

RUN apt-get install -y cmake

RUN mkdir /build/opencv/build \
 && cd /build/opencv/build \
 && cmake -D WITH_CUDA=ON -D OPENCV_EXTRA_MODULES_PATH=/build/opencv_contrib/modules -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local -D ENABLE_CXX11=ON -D BUILD_LIST=calib3d,core,cudaoptflow,cudev,features2d,xfeatures2d .. \
 && make -j \
 && make install

RUN apt-get install -y libproj-dev

# For lensfun
RUN apt-get install -y libglib2.0-dev 
RUN git clone --depth 1 --branch v0.3.95 https://git.code.sf.net/p/lensfun/code /build/lensfun
COPY third_party/lensfun-heapfix.patch /build/lensfun
RUN cd /build/lensfun \
 && git apply lensfun-heapfix.patch
RUN mkdir /build/lensfun/build \
 && cd /build/lensfun/build \
 && cmake -DCMAKE_BUILD_TYPE=Release .. \
 && make -j \
 && make install

#RUN ls > out_ls
#RUN pwd > out_pwd

