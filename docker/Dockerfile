FROM python:3.7.7

WORKDIR /app

COPY . /app
COPY .env.production /app/.env

RUN apt update && apt upgrade -y
RUN apt install libsndfile1 ffmpeg gcc g++ -y sox libboost-all-dev swig git
RUN pip install tensorflow==2.2.0rc2
RUN /app/setup.sh

EXPOSE 5000

CMD ["flask", "run", "--host", "0.0.0.0"]
