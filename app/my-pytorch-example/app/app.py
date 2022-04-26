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

# Preprocessing steps for the image
image_transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

model_file = '/opt/ml/model'
model = torch.jit.load(model_file)

# Put model in evaluation mode for inferencing
model.eval()


def lambda_handler(event, context):

    # Run detect.py
    os.system('python detect.py --weights "best.pt"  --source "./data/images/bus.jpg"')

    ls_output = subprocess.run(["ls", "-l", "./runs/detect/"], stdout=subprocess.PIPE, text=True, input="Hello from the other side")
    print(ls_output.stdout)  # Hello from the other side

    # image_bytes = event['body'].encode('utf-8')
    image = Image.open('./runs/detect/exp/bus.jpg')
    imgByteArr = io.BytesIO()
    image.save(imgByteArr, format='PNG') 

    return {
        'headers': { "Content-Type": "image/jpg" },
        'statusCode': 200,
        'body': base64.b64encode(imgByteArr.getvalue()).decode('utf-8'),
        'isBase64Encoded': True
    }


