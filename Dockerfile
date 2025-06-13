FROM python:3.10-slim

WORKDIR /app

# Copy with proper permissions
COPY perfume_haven/static/ /app/static/
COPY perfume_haven/templates/ /app/templates/
COPY notebooks/perfumes_dataset.csv /app/notebooks/perfumes_dataset.csv
COPY perfume_haven/app.py perfume_haven/requirements.txt /app/

# Explicitly verify static files
RUN ls -la /app/static

RUN pip install -r requirements.txt

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "120", "-k", "uvicorn.workers.UvicornWorker", "app:app"]