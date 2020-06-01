FROM tensorflow/tensorflow:nightly

WORKDIR /app

COPY . /app
COPY .env.production /app/.env

RUN pip install flask flask-cors flask-socketio python-dotenv
RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y wget curl cmake make libsndfile1 ffmpeg gcc g++ swig git
RUN /app/setup.sh asr
RUN rm -r /app/beamsearch_with_lm

EXPOSE 5000

CMD ["flask", "run", "--host", "0.0.0.0"]
