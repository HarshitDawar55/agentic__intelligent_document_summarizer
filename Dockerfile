FROM python:3.10-slim

RUN apt update -y && python -m pip install pip --upgrade

COPY ./requirements.txt /

RUN pip3 install -r requirements.txt -U

COPY ./main.py ./tools.py /

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]