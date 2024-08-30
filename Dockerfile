FROM python:3.7-slim

WORKDIR /src

COPY requirements.txt .
COPY src/ ./src

RUN pip install -r requirements.txt

COPY . .

CMD ["python", "src/main.py"]