FROM python:3.10.12
WORKDIR /AI_SERVER
COPY ./requirements.txt /AI_SERVER/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /AI_SERVER/requirements.txt
COPY ./app /AI_SERVER/app
COPY ./label /AI_SERVER/label
COPY models /AI_SERVER/models
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]