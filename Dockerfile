# syntax=docker/dockerfile:1
FROM node:current-buster
WORKDIR /opt/bitsieve
RUN apt-get update
RUN apt -y install python3 build-essential build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libsqlite3-dev libreadline-dev libffi-dev curl libbz2-dev -y wget
RUN wget https://www.python.org/ftp/python/3.8.12/Python-3.8.12.tar.xz
RUN tar -xf Python-3.8.12.tar.xz
#RUN mv Python3.8.12 /opt/Python3.8.12
RUN cd /opt/Python-3.8.12/
RUN ./configure --enable-optimizations --enable-shared
RUN make
RUN make install
COPY dist/engine /opt/bitsieve
RUN file /opt/bitsieve
RUN ln -s /opt/bitsieve /usr/local/bin/engine
RUN file /opt/bitsieve/engine
COPY entry.sh /usr/bin/entry.sh
RUN chmod +x /usr/bin/entry.sh
#R#UN chmod +x /usr/local/bin/engine
RUN echo "Installing depends from apt"
CMD ["/usr/bin/entry.sh"]
