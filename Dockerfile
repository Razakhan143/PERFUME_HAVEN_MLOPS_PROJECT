FROM python:3.10-slim

WORKDIR /app

COPY perfume_haven/ /app/

COPY notebooks/perfumes_dataset.csv /app/notebooks/perfumes_dataset.csv

RUN pip install -r requirements.txt
EXPOSE 5000

#local
# CMD ["python", "app.py"]  

#Prod

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "120", "app:app"]