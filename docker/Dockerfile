FROM python:3.7.4

ARG model_file=/tmp/model.tf
ARG config_file=/tmp/config.py

WORKDIR /app

COPY .env.production /app/.env
COPY . /app

RUN pip install -r requirements.txt
RUN pip install tensorflow
RUN mkdir /app/assets
RUN [ -f $model_file ] && cp $model_file /app/assets/
COPY $config_file /app/assets/config.py

EXPOSE 5000

CMD ["flask", "run"]
