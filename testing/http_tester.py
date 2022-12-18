import time
import requests

time.sleep(10)

r = requests.get('http://localhost:1318/')
r2 = requests.post('http://localhost:1318/message/',
                   json={'state': 'LANE_FOLLOWING', "direction": "FORWARD"})
