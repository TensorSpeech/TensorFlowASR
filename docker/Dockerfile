FROM python:3.7.7

WORKDIR /app

COPY . /app
COPY .env.production /app/.env

RUN apt update && apt upgrade -y
RUN apt install -y wget cmake make libsndfile1 ffmpeg gcc g++ sox libboost-all-dev swig git
RUN /app/setup.sh

EXPOSE 5000

CMD ["flask", "run", "--host", "0.0.0.0"]
