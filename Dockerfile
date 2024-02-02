FROM dustynv/nanoowl:r35.3.1

# 0. System Requirements
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=en_US.UTF-8 LANGUAGE=en_US:en LC_ALL=en_US.UTF-8
ENV NVIDIA_VISIBLE_DEVICES all
RUN apt-get update && apt-get install -y locales && locale-gen en_US.UTF-8
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y htop tmux gedit libgtk2.0-dev pkg-config
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
RUN apt-get install -y git-lfs && git lfs install

# 1. Install the dependencies
RUN apt-get install python3-pip -y
RUN python3 -m pip install --upgrade pip
RUN pip3 install git+https://github.com/openai/CLIP.git
RUN pip3 install pillow --upgrade
# RUN cd /opt/ && git clone https://github.com/NVIDIA-AI-IOT/torch2trt && \
#     cd torch2trt && python setup.py install --plugins
# RUN cd /opt/ && git clone https://github.com/NVIDIA-AI-IOT/trt_pose && \
#     cd trt_pose && python3 setup.py develop --user
RUN pip3 install matplotlib transformers timm gdown opencv-python opencv-contrib-python
RUN pip3 uninstall opencv

# 2. [Optional] Install the NanoSAM Python package
# RUN cd /opt/ && git clone https://github.com/NVIDIA-AI-IOT/nanosam && \
#     cd nanosam && \
#     git checkout main && mkdir data && \
#     python3 setup.py develop --user

# 3. [Optional] Efficient ViT
# RUN cd /opt/ && git clone --branch dev-python-3.8 https://github.com/tokk-nv/efficientvit.git && \
#     cd efficientvit && \
#     sed 's|^torch.*||g' -i requirements.txt && \
#     sed 's|^opencv-python||g' -i requirements.txt && \
#     sed 's|^transformers||g' -i requirements.txt && \
#     sed 's|^onnx$||g' -i requirements.txt && \
#     sed 's|^onnxruntime$||g' -i requirements.txt && \
#     sed 's|^git+https://github.com/facebookresearch/segment-anything.git||g' -i requirements.txt && \
#     cat requirements.txt && \
#     pip3 install -r requirements.txt 

# 4. Prepare the alternative resources
RUN mkdir /workspace/resources && cd /workspace/resources && \
    gdown https://drive.google.com/uc?id=14-SsvoaTl-esC3JOzomHDnI9OGgdO2OR && \
    wget --quiet --show-progress --progress=bar:force:noscroll --no-check-certificate \
	 https://nvidia.box.com/shared/static/ho09o7ohgp7lsqe0tcxqu5gs2ddojbis.onnx \
	 -O /workspace/resources/mobile_sam_mask_decoder.onnx
RUN git clone https://huggingface.co/google/owlvit-base-patch32
RUN git clone https://huggingface.co/google/owlv2-base-patch16-ensemble


WORKDIR /workspace




 
