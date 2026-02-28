FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel

WORKDIR /
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY handler.py .

CMD [ "python", "-u", "/handler.py" ]