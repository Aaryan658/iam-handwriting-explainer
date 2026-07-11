import requests
import urllib3
urllib3.disable_warnings()

URL = "https://aaryandharrmik-iam-handwriting-explainer.hf.space/"
response = requests.get(URL, verify=False)
html = response.text
import re
match = re.search(r'gradio(?:_api)?_info.*?version":"(.*?)"', html)
if match:
    print("Gradio version:", match.group(1))
else:
    print(html[:500])
