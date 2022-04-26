FROM python:3.8.13-buster
RUN apt update && apt install -y zip htop screen libgl1-mesa-glx \
  g++ \
  make \
  cmake \
  unzip \
  libcurl4-openssl-dev
  
RUN mkdir -p app
COPY bus.jpg /app/data/images/bus.jpg
WORKDIR /app
COPY requirements.txt .
RUN python3 -m pip install -r requirements.txt -t .
COPY . .
COPY model /opt/ml/model
COPY best.pt ./
ADD aws-lambda-rie /usr/local/bin/aws-lambda-rie

RUN pip install \
        --target . \
        awslambdaric

COPY ./entry_script.sh ./entry_script.sh
ENTRYPOINT [ "/app/entry_script.sh" ]

# ENTRYPOINT [ "/usr/local/bin/python", "-m", "awslambdaric" ]
CMD ["app.lambda_handler"]
