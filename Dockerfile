FROM python:3.12-slim

WORKDIR /app

COPY ModelAPI/requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY ModelAPI/ ./ModelAPI/

RUN mkdir -p /app/models

EXPOSE 3000

# Change to the ModelAPI directory
WORKDIR /app/ModelAPI

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "3000"]
