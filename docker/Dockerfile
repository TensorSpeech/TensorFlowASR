FROM python:3.7.4

ARG model_file=/tmp/model.tf

WORKDIR /app

COPY . /app
COPY .env.production /app/.env

RUN apt update && apt upgrade -y
RUN apt install libsndfile1 ffmpeg gcc g++ -y
RUN pip install -r requirements.txt
RUN pip install tensorflow==2.1.0
RUN mkdir /app/assets
RUN test -f $model_file && cp $model_file /app/assets/ || echo "Not copying model file"

EXPOSE 5000

CMD ["flask", "run", "--host", "0.0.0.0"]
