import requests


class LLMClient:

    def __init__(self, model="qwen2.5:7b-instruct", url="http://localhost:11434/api/generate"):
        self.model = model
        self.url = url

    def generate(self, prompt):

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }

        r = requests.post(self.url, json=payload)

        r.raise_for_status()

        return r.json()["response"]