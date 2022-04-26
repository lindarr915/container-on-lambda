FROM python:3.8.13-buster
RUN apt update && apt install -y zip htop screen libgl1-mesa-glx \
  g++ \
  make \
  cmake \
  unzip \
  libcurl4-openssl-dev
  
RUN mkdir -p app
COPY . ./app
COPY model /opt/ml/model
# COPY best.pt ./
COPY bus.jpg ./app/data/images/bus.jpg
WORKDIR /app
RUN python3 -m pip install -r requirements.txt -t .
RUN pip install \
        --target . \
        awslambdaric

ENTRYPOINT [ "/usr/local/bin/python", "-m", "awslambdaric" ]
CMD ["app.lambda_handler"]
