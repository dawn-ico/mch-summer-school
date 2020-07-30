# Build with: docker build . -t mch-summer-school-image
# Run with: docker run -it mch-summer-school-image

FROM ubuntu
COPY ./docker-install-spack.sh /
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt-get install git -y
RUN git config --global url."https://github.com/".insteadOf git@github.com:
RUN apt-get install vim -y
RUN apt-get install python3 -y
RUN apt-get install g++-8 -y
RUN apt-get install curl -y
RUN apt-get install make -y
RUN apt-get install cmake -y
RUN apt-get install unzip -y
RUN apt-get install python3-pip -y
RUN apt-get install gfortran-8 -y
RUN apt-get install libnetcdf-dev -y
RUN apt-get install libnetcdf-c++4-dev -y
RUN apt-get install paraview -y
RUN pip3 install pyyaml
RUN chmod +x /docker-install-spack.sh && /docker-install-spack.sh
WORKDIR /home/root
ENV PATH="/home/root/spack/bin:${PATH}"
RUN git clone https://github.com/dawn-ico/mch-summer-school
WORKDIR /home/root/mch-summer-school
RUN spack dev-build mch-summer-school@master
RUN apt-get install clang-format -y 
