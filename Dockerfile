FROM python:3.8-slim

COPY . /Code

WORKDIR /Code

RUN pip install -r requirements.txt

CMD ["python","GreenPrediction.py"]