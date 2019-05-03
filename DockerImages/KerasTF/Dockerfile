FROM nvcr.io/nvidia/tensorflow:18.01-py3
RUN apt-get -y update && pip install keras
RUN pip install matplotlib
RUN pip install pydot
RUN apt-get -y install python-pydot python-pydot-ng graphviz
RUN apt-get -y install python3.5-tk
RUN git clone https://github.com/jonathanhhb/DataScience.git
