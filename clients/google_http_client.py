import os
import base64
import json
import requests
import google.auth
import google.auth.transport.requests
from google.oauth2 import service_account

class GoogleHTTPClient:
  def __init__(self, model: str ='gemini-1.5-pro-preview-0409', max_retries: int = 5):
    self.model = model
    self.max_retries = max_retries
    self.project = 'playtest-419620'
    self.region = 'us-central1'

    self.access_token = self._get_access_token()

  def request_message(self, messages: list[object], retries: int = 0) -> str:
    url = f"https://{self.region}-aiplatform.googleapis.com/v1/projects/{self.project}/locations/{self.region}/publishers/google/models/{self.model}:generateContent"
    
    headers = {
      'Authorization': f'Bearer {self.access_token}',
      'Content-Type': 'application/json',
    }

    data = json.dumps({ 
      'contents': {
        'role': 'user',
        'parts': [messages]
      } 
    })

    response = requests.post(url, headers=headers, data=data)
    response = json.loads(response.text)

    try:
      response = response['candidates'][0]['content']['parts'][0]['text'].strip()
      return response
    except Exception as error:
      if retries == self.max_retries:
        raise Exception(f'Failed to fetch from google:', error)

      return self.request_message(messages, retries + 1)
  
  def format_image_message(self, image: bytes) -> object:
    return {
      "inlineData": {
        "mimeType": 'image/png',
        "data": base64.b64encode(image).decode('utf-8')
      }
    }
  
  def format_text_message(self, text: str) -> object:
    return { 'text': text }

  def _get_access_token(self) -> str:

    service_account_key = './google_credentials.json'
    credentials = service_account.Credentials.from_service_account_file(
        service_account_key, 
        scopes=['https://www.googleapis.com/auth/cloud-platform'] 
    )

    auth_req = google.auth.transport.requests.Request()
    credentials.refresh(auth_req)

    return credentials.token
