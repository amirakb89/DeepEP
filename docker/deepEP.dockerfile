FROM docker.io/rocm/pytorch:rocm6.3.4_ubuntu24.04_py3.12_pytorch_release_2.4.0

RUN apt update && apt install -y openssh-server iputils-ping build-essential gfortran flex libtool autoconf automake \
                 python3 python3-pip python3-venv perl m4 pkg-config libevent-dev libhwloc-dev netcat-openbsd libibverbs-dev ibverbs-utils

RUN mkdir /workspace

WORKDIR /workspace

RUN git clone https://github.com/ROCm/rocSHMEM.git -b develop

RUN chmod +x /workspace/rocSHMEM/scripts/install_dependencies.sh
SHELL ["/bin/bash", "-lc"]
RUN BUILD_DIR=/workspace /workspace/rocSHMEM/scripts/install_dependencies.sh


ENV PATH="/workspace/install/ompi/bin:/workspace/install/ompi/sbin:${PATH}" \
    LD_LIBRARY_PATH="/workspace/install/ompi/lib:${LD_LIBRARY_PATH}" \
    C_INCLUDE_PATH="/workspace/install/ompi/include:${C_INCLUDE_PATH}" \
    CPLUS_INCLUDE_PATH="/workspace/install/ompi/include:${CPLUS_INCLUDE_PATH}" \
    PKG_CONFIG_PATH="/workspace/install/ompi/lib/pkgconfig:${PKG_CONFIG_PATH}" \
    MANPATH="/workspace/install/ompi/share/man:${MANPATH}" \
    CMAKE_C_COMPILER="/workspace/install/ompi/bin/mpicc" \
    CMAKE_CXX_COMPILER="/workspace/install/ompi/bin/mpicxx"

WORKDIR /workspace/rocSHMEM
RUN mkdir build
RUN cd build && \
../scripts/build_configs/gda_mlx5 -DGDA_BNXT=ON -DUSE_IPC=ON


RUN pip install setuptools==75.1.0

# Create and apply the patch
RUN echo '--- /opt/conda/envs/py_3.12/lib/python3.12/site-packages/torch/utils/cpp_extension.py' > /tmp/torch.patch && \
    echo '+++ /opt/conda/envs/py_3.12/lib/python3.12/site-packages/torch/utils/cpp_extension.py' >> /tmp/torch.patch && \
    echo '@@ -2114,7 +2114,7 @@' >> /tmp/torch.patch && \
    echo "     if cflags is not None:" >> /tmp/torch.patch && \
    echo "         for flag in cflags:" >> /tmp/torch.patch && \
    echo "             if 'amdgpu-target' in flag or 'offload-arch' in flag:" >> /tmp/torch.patch && \
    echo "-                return ['-fno-gpu-rdc']" >> /tmp/torch.patch && \
    echo "+                return ['-fgpu-rdc']" >> /tmp/torch.patch && \
    echo "     # Use same defaults as used for building PyTorch" >> /tmp/torch.patch && \
    echo "     # Allow env var to override, just like during initial cmake build." >> /tmp/torch.patch && \
    echo "     _archs = os.environ.get('PYTORCH_ROCM_ARCH', None)" >> /tmp/torch.patch && \
    echo "@@ -2127,7 +2127,7 @@" >> /tmp/torch.patch && \
    echo "     else:" >> /tmp/torch.patch && \
    echo "         archs = _archs.replace(' ', ';').split(';')" >> /tmp/torch.patch && \
    echo "     flags = [f'--offload-arch={arch}' for arch in archs]" >> /tmp/torch.patch && \
    echo "-    flags += ['-fno-gpu-rdc']" >> /tmp/torch.patch && \
    echo "+    flags += ['-fgpu-rdc']" >> /tmp/torch.patch && \
    echo "     return flags" >> /tmp/torch.patch && \
    echo "" >> /tmp/torch.patch && \
    echo " def _get_build_directory(name: str, verbose: bool) -> str:" >> /tmp/torch.patch && \
    patch /opt/conda/envs/py_3.12/lib/python3.12/site-packages/torch/utils/cpp_extension.py < /tmp/torch.patch
ARG REBUILD_FROM=0
WORKDIR /workspace

COPY /DeepEP /workspace/DeepEP

# WORKDIR /workspace/DeepEP
# RUN OMPI_DIR=/workspace/install/ompi ROCSHMEM_DIR=/root/rocshmem python setup.py --variant rocm build develop


WORKDIR /workspace/DeepEP/
