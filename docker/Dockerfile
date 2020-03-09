FROM python:3.7.4

ARG model_file=/tmp/model.tf
ARG config_file=/tmp/config.py

WORKDIR /app

COPY .env.production /app/.env
COPY . /app

RUN apt update && apt upgrade -y
RUN apt install libsndfile1 ffmpeg gcc g++ -y
RUN pip install -r requirements.txt
RUN pip install tensorflow==2.1.0
RUN mkdir /app/assets
RUN test -f $model_file && cp $model_file /app/assets/ || echo "Not copying model file"
COPY $config_file /app/assets/config.py

EXPOSE 5000

CMD ["flask", "run"]
