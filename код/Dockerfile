FROM python:3.11.4

ENV PYTHONUNBUFFERED=1

WORKDIR /usr/src/app 

COPY . .

RUN pip install -r requirements.txt

CMD [ "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000" ]