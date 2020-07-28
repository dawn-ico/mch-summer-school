# This image is intended for testing the spack installation of icondusk-e2e.
# It can also serve as an installation tutorial for Ubuntu users.
# As this is currently a private repo, it requires your private key as argument. 

# Build with: docker build . --build-arg SSH_PRIVATE_KEY="$(cat ~/.ssh/id_rsa)" -t icondusk-e2e-image
# Run with: docker run -it icondusk-e2e-image

FROM ubuntu
COPY ./docker-install-spack.sh /
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt-get install git -y
RUN git config --global url."https://github.com/".insteadOf git@github.com:
RUN apt-get install python3 -y
RUN apt-get install g++-8 -y
RUN apt-get install curl -y
RUN apt-get install make -y
RUN apt-get install cmake -y
RUN apt-get install unzip -y
RUN apt-get install python3-pip -y
RUN apt-get install gfortran-8 -y
RUN pip3 install pyyaml
RUN chmod +x /docker-install-spack.sh && /docker-install-spack.sh
WORKDIR /home/root
ENV PATH="/home/root/spack/bin:${PATH}"
RUN spack -C /home/root/ install mch-summer-school@master%gcc@8.4.0; exit 0
CMD /bin/bash