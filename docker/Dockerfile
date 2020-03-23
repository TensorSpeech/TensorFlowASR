FROM python:3.7.7

WORKDIR /app

COPY . /app
COPY .env.production /app/.env

RUN apt update && apt upgrade -y
RUN apt install libsndfile1 ffmpeg gcc g++ -y
RUN pip install -r requirements.txt
RUN pip install tensorflow==2.1.0
RUN mkdir /app/assets
COPY export_dir /app/assets/export_dir

EXPOSE 5000

CMD ["flask", "run", "--host", "0.0.0.0"]
