FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

RUN useradd -m -u 1000 user
USER user
WORKDIR $HOME/app

COPY --chown=user requirements-space.txt ./requirements-space.txt
RUN python -m pip install --upgrade pip && python -m pip install -r requirements-space.txt

COPY --chown=user egocentric_dataset_test/__init__.py ./egocentric_dataset_test/__init__.py
COPY --chown=user egocentric_dataset_test/competition ./egocentric_dataset_test/competition
COPY --chown=user server ./server
COPY --chown=user inference.py ./inference.py
COPY --chown=user openenv.yaml ./openenv.yaml
COPY --chown=user README.md ./README.md
COPY --chown=user pyproject.toml ./pyproject.toml
COPY --chown=user uv.lock ./uv.lock

EXPOSE 7860

CMD ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
