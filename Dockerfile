FROM python:3.10-slim

WORKDIR /app

# Copy with proper permissions
COPY perfume_haven/static/ /app/perfume_haven/static/
COPY perfume_haven/templates/ /app/perfume_haven/templates/
COPY notebooks/perfumes_dataset.csv /app/notebooks/perfumes_dataset.csv
COPY perfume_haven/app.py perfume_haven/requirements.txt /app/

# Explicitly verify static files
RUN ls -la /app/perfume_haven/static

RUN pip install -r requirements.txt

EXPOSE 5000
# for local use
# CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000", "--reload", "--log-level", "debug"]

# for production use
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "120", "-k", "uvicorn.workers.UvicornWorker", "app:app"]