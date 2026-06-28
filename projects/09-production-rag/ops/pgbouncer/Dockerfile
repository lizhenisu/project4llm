FROM debian:trixie-slim

RUN apt-get -o Acquire::Retries=3 update \
    && DEBIAN_FRONTEND=noninteractive apt-get -o Acquire::Retries=3 install -y --no-install-recommends \
        pgbouncer \
        postgresql-client \
        python3-minimal \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --create-home --home-dir /var/lib/pgbouncer pgbouncer \
    && install -d -o pgbouncer -g pgbouncer /var/run/pgbouncer-config

COPY start_pgbouncer.py /usr/local/bin/start_pgbouncer.py

USER pgbouncer

EXPOSE 6432

ENTRYPOINT ["python3", "/usr/local/bin/start_pgbouncer.py"]
