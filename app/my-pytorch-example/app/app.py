import torch
import torchvision
import base64
import json
import numpy as np

from PIL import Image
from io import BytesIO

import os, io

# import detect
import subprocess
import boto3 

# Preprocessing steps for the image
image_transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

model_file = '/opt/ml/model'
model = torch.jit.load(model_file)

# Put model in evaluation mode for inferencing
model.eval()


def lambda_handler(event, context):

    bucket_name = os.environ["BUCKET_NAME"] 
    object_key = json.loads(event["body"])["object_key"]

    s3 = boto3.client("s3")
    s3.download_file(bucket_name, object_key, "/tmp/hello.jpg")
    
    # Run detect.py
    os.system('python detect.py --weights "best.pt"  --source "/tmp/hello.jpg"')

    ls_output = subprocess.run(["ls", "-l", "/tmp/runs/detect/"], stdout=subprocess.PIPE, text=True, input="Hello from the other side")
    print(ls_output.stdout)  # Hello from the other side

    # image_bytes = event['body'].encode('utf-8')
    image = Image.open('/tmp/runs/detect/exp/hello.jpg')
    imgByteArr = io.BytesIO()
    image.save(imgByteArr, format='JPEG') 

    delete_file = subprocess.run(["rm", "-rf", "/tmp/runs/detect/"], stdout=subprocess.PIPE, text=True, input="Delete")

    return {
        'headers': { "Content-Type": "image/jpg" },
        'statusCode': 200,
        'body': base64.b64encode(imgByteArr.getvalue()).decode('utf-8'),
        'isBase64Encoded': True
    }


