FROM python:3.11-slim
RUN apt-get update && apt-get install -y \
    curl \
    jq \
    && rm -rf /var/lib/apt/lists/*
# 設定工作目錄
WORKDIR /app

# Dockerfile 位於 rag_server 中，
# 首先複製 requirements.txt 並安裝
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir numpy==1.26.4
# 複製 rag_server 資料夾中的所有其他檔案 (ingest.py, server.py 等)到容器的 /app 工作目錄中
COPY . .

