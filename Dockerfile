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
RUN unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy; \
    apt-get -o Acquire::Retries=5 update \
    && apt-get -o Acquire::Retries=5 install -y --no-install-recommends ca-certificates curl ffmpeg git gh nodejs npm \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir "git+https://github.com/Panniantong/Agent-Reach.git@1221ecd0c3e0502ee37406f03543bedf7503f2c7" bilibili-cli twitter-cli "git+https://github.com/public-clis/rdt-cli.git@5e4fb3720d5c174e976cd425ccc3b879d52cac66" \
    && npm install -g mcporter
COPY assistant ./assistant
COPY app.py ./
COPY oauth_server.py ./
COPY skills ./skills
COPY deploy/agent-reach-entrypoint.sh /usr/local/bin/agent-reach-entrypoint
RUN chmod +x /usr/local/bin/agent-reach-entrypoint

CMD ["agent-reach-entrypoint"]
