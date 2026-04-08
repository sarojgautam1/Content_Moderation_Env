import subprocess
import requests
import time

p = subprocess.Popen(["python", "-m", "uvicorn", "server.app:app", "--port", "8008"])
time.sleep(2)

try:
    r = requests.get("http://127.0.0.1:8008/")
    print("/ status:", r.status_code)
    print("/ text:", r.text)

    r = requests.get("http://127.0.0.1:8008/tasks")
    print("/tasks status:", r.status_code)
    print("/tasks text:", r.text)
    
    r = requests.get("http://127.0.0.1:8008/state")
    print("/state status:", r.status_code)
    print("/state text:", r.text)
except Exception as e:
    print("Error:", e)
finally:
    p.terminate()
