FROM python:3.6-alpine
ADD requirement.txt api.py identity_tokenizer.py vectorize.pkl model.pkl /srv/
WORKDIR /srv

RUN apk add --virtual --no-cache build-runtime \
    build-base python-dev openblas-dev freetype-dev pkgconfig gfortran \
    && ln -s /usr/include/locale.h /usr/include/xlocale.h \
    && pip install --upgrade pip \
    && pip install -r requirement.txt \
    && apk del build-runtime \
    && apk add --no-cache libstdc++ openblas

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "api:app"]
