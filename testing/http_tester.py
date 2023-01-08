import time
import requests

import io
from PIL import Image


r = requests.get('http://192.168.0.108:1318/image')
image = Image.open(io.BytesIO(r.content)).convert("RGB")
image.show()

# r2 = requests.post('http://localhost:1318/message/',
#                    json={'state': 'LANE_FOLLOWING', "direction": "FORWARD"})
