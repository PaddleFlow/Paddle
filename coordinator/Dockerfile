FROM python:3.7
WORKDIR /home/antman
COPY . .

RUN apt update && \
  pip install -r requirements.txt

CMD python local/local_coordinator.py
