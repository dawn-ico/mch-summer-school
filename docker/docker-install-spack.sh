#!/bin/bash
set -e

cd /home
mkdir root
cd root
git clone https://github.com/spack/spack.git
cd spack/bin
./spack install libelf
export PATH=$PATH:/home/root/spack/bin
cd ..
source share/spack/setup-env.sh
spack compiler find
cd /home/root
git clone https://github.com/mroethlin/spack-mch
cd spack-mch
git checkout summer-school
spack repo add .
cd ..
echo "config:" >> config.yaml
echo "  connect_timeout: 30" >> config.yaml