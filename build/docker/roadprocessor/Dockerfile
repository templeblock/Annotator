####################################################################################################################
FROM roads-cppbuild
COPY . /build/
WORKDIR /build
RUN tundra2 linux-clang-release RoadProcessor

####################################################################################################################
#FROM ubuntu:18.04
#
#RUN apt-get update
#RUN apt-get install -y software-properties-common
#RUN apt-add-repository ppa:graphics-drivers/ppa
#RUN apt-get install -y nvidia-driver-396
#
##RUN apt-get update
##RUN apt-cache search nvidia
##RUN apt-get install -y nvidia-396

####################################################################################################################
# cudnn7-devel is 1 GB
# cudnn7-runtime is 712 MB
# Just using -devel- for now, because our internet is slow in the building
#FROM nvidia/cuda:9.2-cudnn7-devel-ubuntu18.04
FROM nvidia/cudagl:9.2-runtime-ubuntu18.04

RUN rm -rf /var/lib/apt/lists/* && apt-get update
RUN apt-get install -y \
 libomp5 \
 libproj12 \
 libpng16-16 \
 libglib2.0-0 \
 libcurl4 \
 libunwind8

#RUN git clone https://github.com/NVIDIA/libglvnd.git

COPY --from=0 \
 /usr/local/lib/libavcodec.so.58.18.100 \
 /usr/local/lib/libavdevice.so.58.3.100 \
 /usr/local/lib/libavfilter.so.7.16.100 \
 /usr/local/lib/libavformat.so.58.12.100 \
 /usr/local/lib/libavutil.so.56.14.100 \
 /usr/local/lib/libjpeg.so.62.3.0 \
 /usr/local/lib/liblensfun.so.0.3.95 \
 /usr/local/lib/libopencv_calib3d.so.3.4.1 \
 /usr/local/lib/libopencv_core.so.3.4.1 \
 /usr/local/lib/libopencv_features2d.so.3.4.1 \
 /usr/local/lib/libopencv_imgproc.so.3.4.1 \
 /usr/local/lib/libopencv_xfeatures2d.so.3.4.1 \
 /usr/local/lib/libswresample.so.3.1.100 \
 /usr/local/lib/libswscale.so.5.1.100 \
 /usr/local/lib/libturbojpeg.so.0.2.0 \
 /usr/local/lib/

# /usr/local/lib/libopencv_cudaarithm.so.3.4.1 \
# /usr/local/lib/libopencv_cudaimgproc.so.3.4.1 \
# /usr/local/lib/libopencv_cudaoptflow.so.3.4.1 \
# /usr/local/lib/libopencv_cudawarping.so.3.4.1 \
# /usr/local/lib/libopencv_cudev.so.3.4.1 \
# /usr/local/lib/libopencv_video.so.3.4.1 \

COPY --from=0 \
 /usr/local/share/lensfun/ \
 /usr/local/share/lensfun/

RUN mkdir /deploy
COPY --from=0 /build/t2-output/linux-clang-release-default/RoadProcessor /deploy/
COPY --from=0 \
 /build/t2-output/linux-clang-release-default/libVideo.so \
 /build/t2-output/linux-clang-release-default/libpal.so \
 /usr/local/lib/

RUN apt-get install -y libxcb-shape0

RUN ldconfig

ENTRYPOINT ["/deploy/RoadProcessor"]