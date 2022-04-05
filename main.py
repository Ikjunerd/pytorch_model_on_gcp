from google.cloud import storage
import requests
from PIL import Image
import numpy as np
import torch
from torchvision import datasets, models, transforms
import torch.nn as nn

def hello_gcs(event, context):
     """Triggered by a change to a Cloud Storage bucket.
     Args:
          event (dict): Event payload.
          context (google.cloud.functions.Context): Metadata for the event.
     """
     file = event #input jpg file
     inputfilename = file['name']
     chatid = inputfilename.split('.')[0] #input filename is chatid
     url = 'https://api.telegram.org/bot[YOUR_BOT_API_KEY]/sendmessage'

     data = {'chat_id': chatid, 'text': '사진 속 동물은...🤔', 'parse_mode': 'Markdown'} 
     res = requests.post(url, data=data)

     device = torch.device('cpu')

     model = models.resnet50(pretrained=False)

     model.fc = nn.Sequential(
          nn.Linear(2048, 128),
          nn.ReLU(),
          nn.Linear(128, 64),
          nn.ReLU(),
          nn.Linear(64, 1)) 

     client = storage.Client()
     bucket = client.get_bucket('YOUR_BUCKET_NAME')
     blob = bucket.get_blob("PYTORCH_MODEL_NAME.pth")
     blob.download_to_filename("/tmp/model.pth") #download pytorch model

     model.load_state_dict(torch.load('/tmp/model.pth',map_location=device))
    
     blob = bucket.get_blob(inputfilename)
     blob.download_to_filename('/tmp/'+inputfilename) #download input jpg file

     testee_img = Image.open('/tmp/'+inputfilename)

     data_transforms = {
          'test': transforms.Compose([
               transforms.Resize((224, 224)),
               transforms.ToTensor(),
               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
          ]),
     }

     testee_input = data_transforms['test'](testee_img)
     testee_batch = torch.stack([testee_input, testee_input])

     testee_batch = testee_batch.to(device)
     output = model(testee_batch)

     decisions = torch.sigmoid(output).cpu().data.numpy()

     prediction = decisions[0,0] > 0.5

     feeling = '😉'
     if decisions[0,0] > 0.6:
          feeling = '😀'
     else: 
          feeling = '🙄'

     text = '몰라요'
     if prediction:
          text = '개입니다. ' + feeling
     else:
          text = '고양이입니다. '  + feeling

     data = {'chat_id': chatid, 'text': text, 'parse_mode': 'Markdown'} 
     res = requests.post(url, data=data)
    