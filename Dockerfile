FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_LINK_MODE=copy PUSULA_HOME=/data
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-install-project --no-dev
COPY src ./src
COPY ledger ./ledger
RUN uv sync --locked --no-dev
# /data: bar store, implied vols, forward books. /app/ledger and /app/docs: the research ledger
# and generated reports; they must outlive the container (a named volume is initialised from
# the image's ledger on first use), or registrations and results are lost with it.
VOLUME ["/data", "/app/ledger", "/app/docs"]
# Batch image (data, study, forward). The dashboard binds 127.0.0.1 by design and is meant to be
# run on the host: `uv run pusula web`.
# nightly: docker run --rm --env-file .env -v pusula-data:/data -v pusula-ledger:/app/ledger \
#          -v pusula-docs:/app/docs pusula forward --family all
ENTRYPOINT ["uv", "run", "--no-dev", "pusula"]
CMD ["--help"]
