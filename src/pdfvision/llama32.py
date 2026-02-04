import os
import requests
import json
from dotenv import load_dotenv
import logging
load_dotenv()

 
def llama32(messages, model_size=11):
  logger = logging.getLogger('__main__.'+__name__)
  try:
    model = os.environ["MODEL"]
    url= os.environ["PSAFINT_API_URL"]

    payload = {
      "model": model,
      "max_tokens": 4096,
      "temperature": 0.0,
      "stop": ["<|eot_id|>","<|eom_id|>"],
      "stream": False,
      "messages": messages     
    }

    headers = {
      "Accept": "application/json",
      "Content-Type": "application/json",
      "Authorization": "Basic " + os.environ["TOKEN"]
    }

    response = requests.request("POST", url, headers=headers, data=json.dumps(payload))
    if response.status_code != 200:
      raise Exception(response.status_code, response.content)
    else:
      res = json.loads(response.text)

    if 'error' in res:
      raise Exception(res['error'])

    return res['choices'][0]['message']['content']

  except Exception as e:
    if response.status_code in range(400, 600):
      if "422" in str(e) and  "max_new_tokens" in str(e):
          return "max_new_token_error"
      else:
        return str(e)