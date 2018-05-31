FROM python:3.6-alpine
MAINTAINER Diane Losardo <dlosardo@gmail.com>

ENV INSTALL_PATH /webapp
RUN mkdir -p $INSTALL_PATH

WORKDIR $INSTALL_PATH

COPY . .
RUN apk add --update curl gcc g++
RUN ln -s /usr/include/locale.h /usr/include/xlocale.h
RUN python setup.py install

RUN pip install -r webapp/requirements.txt

CMD gunicorn -b 0.0.0.0:8000 --access-logfile - "webapp.app:create_app()"
