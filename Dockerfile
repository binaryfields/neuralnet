FROM fedora:27

RUN groupadd --gid 1000 xaos && \
	useradd --uid 1000 --gid 1000 --create-home xaos --shell /usr/bin/fish && \
	dnf install -y fish ncdu sudo && dnf clean all && \
	echo 'xaos ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/xaos

RUN dnf install -y python3-tkinter && dnf clean all

USER xaos
ENV PATH $PATH:/home/xaos/.local/bin

RUN ln -s /srv/app ~/app
RUN pip3 install --user h5py jupyter matplotlib numpy pandas pillow scipy sklearn tensorflow

WORKDIR /srv/app
ENTRYPOINT [ "fish" ]
