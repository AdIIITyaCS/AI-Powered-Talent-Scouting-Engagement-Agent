import os
import requests
from dotenv import load_dotenv

load_dotenv()
key = os.getenv('AFFINDA_API_KEY')
print('KEY OK', bool(key))
headers = {'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'}
payload = {'text': 'Role: Senior AI Agent Developer.'}
urls = [
    'https://api.affinda.com/v1/jobdescription-parser',
    'https://api.affinda.com/jobdescription/v1/parser',
    'https://api.affinda.com/v1/parser',
    'https://api.affinda.com/resume/v1/parser',
    'https://api.affinda.com/v1/recruiter/parser',
]
for url in urls:
    try:
        r = requests.post(url, json=payload, headers=headers, timeout=30)
        print(url, r.status_code, r.text[:200])
    except Exception as e:
        print(url, 'ERROR', e)
