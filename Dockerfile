FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG ALL_PROXY
ENV HTTP_PROXY=${HTTP_PROXY} \
    HTTPS_PROXY=${HTTPS_PROXY} \
    ALL_PROXY=${ALL_PROXY}
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY assistant ./assistant
COPY app.py ./
COPY oauth_server.py ./
COPY skills ./skills

CMD ["python", "app.py"]
