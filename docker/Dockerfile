FROM tensorflow/tensorflow:nightly

WORKDIR /app

COPY . /app
COPY .env.production /app/.env

RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y wget cmake make libsndfile1 ffmpeg gcc g++ git
RUN /app/setup.sh asr

EXPOSE 5000

CMD ["flask", "run", "--host", "0.0.0.0"]
