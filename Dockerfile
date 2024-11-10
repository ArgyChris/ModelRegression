FROM python:3.7.9-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements_app.txt && pip list && uvicorn --version

EXPOSE 8080

CMD ["uvicorn", "application:app", "--host", "0.0.0.0", "--port", "8080"]
