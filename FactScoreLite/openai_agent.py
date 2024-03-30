from openai import OpenAI


class OpenAIAgent:

    def __init__(self):
        self.client = OpenAI()
        self.max_tokens = 1024
        self.temp = 0.7
        self.model_name = "gpt-4-turbo-preview"

    def generate(self, prompt):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=self.temp,
        )

        return response.choices[0].message.content
