import os
import requests
import base64
from dotenv import load_dotenv

class OpenAIClient:
  def __init__(self, model: str = 'gpt-4o', system: str | None = None, max_tokens: int = 1024, temperature: float = 0.0, max_retries: int = 5, response_format: str | None = None):
    self.model = model
    self.system_message = { 'role': 'system', 'content': system } if system else None
    self.max_tokens = max_tokens
    self.temperature = temperature
    self.max_retries = max_retries
    self.response_format = response_format

    load_dotenv()
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    
    self.headers = {
      'Content-Type': 'application/json',
      'Authorization': f'Bearer {OPENAI_API_KEY}'
    }

  def format_image_message(self, image: bytes) -> object:
    return {
      'role': 'user',
      'content': [{
        'type': 'image_url',
        'image_url': {
          'url': f"data:image/png;base64,{base64.b64encode(image).decode('utf-8')}",
          'detail': 'high'
        }
      }]
    }
  
  def format_text_message(self, text: str) -> object:
    return { 'role': 'user', 'content': text }

  def request_message(self, messages: list[object], retries: int = 0) -> str:
    payload = {
      'model': self.model,
      'max_tokens': self.max_tokens,
      'temperature': self.temperature,
      'messages': [self.system_message] + messages if self.system_message else messages,
    }

    if self.response_format is not None:
      payload['response_format'] = { "type": self.response_format }

    response = requests.post(
      'https://api.openai.com/v1/chat/completions', 
      headers=self.headers,
      json=payload
    )

    response = response.json()

    if 'choices' in response:
      content = response['choices'][0]['message']['content']
      return content
    else:
      if retries >= self.max_retries:
        print(response)
        raise Exception(f'Failed to fetch from OpenAI API')
      
      return self.request_message(messages, retries + 1)