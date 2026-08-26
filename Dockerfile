# Evidence RAG
#
# Single-service Flask app serving both the API and the static frontend.
# Shell-form ENTRYPOINT (not exec/JSON-array form) deliberately - exec
# form can't expand environment variables, and this needs to respect
# whatever $PORT a hosting platform assigns (learned this the hard way
# on an earlier project in this portfolio, where a hardcoded port in
# exec form would have silently ignored Render's actual port assignment).

FROM python:3.12-slim
WORKDIR /app

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ .
COPY static/ ../static/
COPY templates/ ../templates/

EXPOSE 5004
ENTRYPOINT gunicorn --workers 1 --threads 4 --bind 0.0.0.0:${PORT:-5004} --timeout 120 run:app
