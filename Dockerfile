FROM python:3.6-alpine
ADD requirement.txt api.py identity_tokenizer.py /srv/
WORKDIR /srv

RUN apk add --virtual build-runtime \
    build-base python-dev openblas-dev freetype-dev pkgconfig gfortran \
    && ln -s /usr/include/locale.h /usr/include/xlocale.h \
    && pip install --upgrade pip \
    && pip install -r requirement.txt \
    && apk del build-runtime \
    && rm -rf /var/cache/apk/*

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "api:app"]
