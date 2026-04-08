FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app/openenv_quant

ENV PYTHONPATH=/app
ENV PORT=7860

EXPOSE 7860

CMD ["uvicorn", "openenv_quant.app:app", "--host", "0.0.0.0", "--port", "7860"]
