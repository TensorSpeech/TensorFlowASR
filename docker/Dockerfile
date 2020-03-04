FROM python:3.7.4

ARG model_file=/tmp/model.tf

WORKDIR /app

COPY .env.production /app/.env
COPY . /app

RUN pip install -r requirements.txt
RUN pip install tensorflow
RUN mkdir /app/assets
COPY $model_file /app/assets/

EXPOSE 5000

CMD ["flask", "run"]