FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

WORKDIR /app

COPY ctools /app/ctools
COPY mri-requirements.txt /app/mri-requirements.txt
RUN pip install --no-cache-dir -r /app/mri-requirements.txt

COPY bronco /app/bronco
COPY calculate_vesselness_stats.py /app/calculate_vesselness_stats.py
COPY mri_vessel_reporting_config.example.json /app/mri_vessel_reporting_config.example.json

ENTRYPOINT ["python", "/app/calculate_vesselness_stats.py"]
CMD ["--help"]
