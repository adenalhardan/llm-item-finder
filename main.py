import base64
from object_detector import ObjectDetector

with open('image.jpg', 'rb') as image_file:
  image = base64.b64encode(image_file.read()).decode('utf-8')

object_detector = ObjectDetector()
print(object_detector.detect_objects(image))