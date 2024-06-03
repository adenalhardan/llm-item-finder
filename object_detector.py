import base64
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import itertools
import random
from multiprocessing.pool import ThreadPool
from vertexai.generative_models import Part
from clients.google_http_client import GoogleHTTPClient
from clients.openai_client import OpenAIClient

class ObjectDetector:
  def __init__(self, chunk_threads: int = 100, item_threads: int = 100, cuts: list[tuple[int, int]] = [(4, 3)]):
    with open('prompts/identifier_prompt.txt', 'r') as file:
      self.identifier_prompt = file.read()

    with open('prompts/locator_prompt.txt', 'r') as file:
      self.locator_prompt = file.read()

    with open('prompts/verifier_prompt.txt', 'r') as file:
      self.verifier_prompt = file.read()

    self.cuts = cuts

    self.google_client = GoogleHTTPClient()
    self.openai_client = OpenAIClient(response_format='json_object')

    self.chunk_pool = ThreadPool(chunk_threads)
    self.item_pool = ThreadPool(item_threads)

  def __del__(self):
    self.chunk_pool.close()
    self.chunk_pool.join()
    self.item_pool.close()
    self.item_pool.join()

  def detect_objects(self, image: str) -> list[object]:
    self.labels, labeled_image = self._label_image(image)

    chunks = list(itertools.chain(*[self._chunkize_image(image, *cut) for cut in self.cuts]))
    labeled_chunks = list(itertools.chain(*[self._chunkize_image(labeled_image, *cut) for cut in self.cuts]))

    items = self.chunk_pool.starmap(self._identifier_worker, zip(chunks, labeled_chunks))
    items = list(itertools.chain(*items))

    random.shuffle(items) # agents pay more attention to items with lower index
    return items

  def _label_image(self, 
    image: str, 
    increment: int = 30, 
    padding: int = 10,
    font_size: float = 11.0,
    font_color: str = 'magenta'
  ) -> tuple[dict[str, tuple[int, int]], str]:
    buffer = BytesIO(base64.b64decode(image))
    image = Image.open(buffer)

    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(font_size)
    width, height = image.size

    labels = {}
    label = '0'

    for x in range(padding, width - padding, increment):
        for y in range(padding, height - padding, increment):
            background = draw.textbbox((x, y), label, font=font)
            draw.rectangle(background, fill=(255, 255, 255, 150))
            draw.text((x, y), label, fill=font_color, font=font)

            labels[label] = (x, y)
            label = str(int(label) + 1)

    image.save('output.png')

    with BytesIO() as buffer:
        image.save(buffer, format='png')
        labeled_image = buffer.getvalue()

    return labels, base64.b64encode(labeled_image).decode('utf-8')
  
  def _chunkize_image(self, 
    image: str, 
    vertical_cuts: int = 4, 
    horizontal_cuts: int = 3
  ) -> list[bytes]:
    buffer = BytesIO(base64.b64decode(image))
    image = Image.open(buffer)
    width, height = image.size

    vertical_step = width // (vertical_cuts + 1)
    horizontal_step = height // (horizontal_cuts + 1)

    chunks = []

    for i in range(horizontal_cuts + 1):
      for j in range(vertical_cuts + 1):
        left = j * vertical_step 
        top = i * horizontal_step
        right = (j + 1) * vertical_step if j != vertical_cuts else width
        bottom = (i + 1) * horizontal_step if i != horizontal_cuts else height
        
        cropped_image = image.crop((left, top, right, bottom))

        with BytesIO() as buffer:
          cropped_image.save(buffer, format="PNG")
          chunk = buffer.getvalue()

        chunks.append(chunk)

    return chunks
  
  def _identifier_worker(self, chunk: str, labeled_chunk: bytes) -> list[object]:
    try:
      chunk_message = self.google_client.format_image_message(chunk)
      labeled_chunk_message = self.google_client.format_image_message(labeled_chunk)
      prompt_message = self.google_client.format_text_message(self.identifier_prompt)

      item_names = self.google_client.request_message([chunk_message, prompt_message])
      item_names = [item_name.strip() for item_name in item_names.split('|')]

      items = self.item_pool.starmap(self._locator_worker, zip(item_names, itertools.repeat(labeled_chunk_message)))
      items = list(filter(None, items))

      items = self.item_pool.starmap(self._verify_worker, zip(items, itertools.repeat(labeled_chunk)))
      items = list(filter(None, items))
  
      return items

    except Exception as error:
      print('identifier worker error: ', error)
      return []
    
  def _locator_worker(self, name: str, labeled_chunk_message: Part) -> object | None:
    try:
      prompt_message = self.google_client.format_text_message(self.locator_prompt.format(name=name))
      label = self.google_client.request_message([labeled_chunk_message, prompt_message])

      if label not in self.labels:
        raise Exception(f'Label {label} not found in labels')
      
      x, y = self.labels[label]
      return { 'name': name, 'label': label, 'x': x, 'y': y }
    
    except Exception as error:
      print('locator worker error:', error)
      return None
    
  def _verify_worker(self, item: object, labeled_chunk: bytes) -> object | None:
    try:
      labeled_chunk_message = self.openai_client.format_image_message(labeled_chunk)
      prompt = self.verifier_prompt.format(name=item['name'], label=item['label'])
      prompt_message = self.openai_client.format_text_message(prompt)

      response = self.openai_client.request_message([labeled_chunk_message, prompt_message])
      _, verified = [part.strip() for part in response.split('|')]

      match verified:
        case 'yes': return item
        case 'no': return None
        case _: return None

    except Exception as error:
      print('verify worker error:', error)
      return item