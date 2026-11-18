# 智铺 ShopMind API 镜像
# 构建（仓库根目录）：docker build -f deploy/Dockerfile -t shopmind-api .
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

WORKDIR /srv/shopmind

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend ./backend
COPY scripts ./scripts
COPY eval ./eval
COPY frontend ./frontend

EXPOSE 8000

# 容器内默认离线 Mock；生产通过环境变量注入 LLM/Milvus/MySQL/Redis 配置
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
