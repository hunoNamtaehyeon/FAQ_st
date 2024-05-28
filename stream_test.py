import openai
import os
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv("/mnt/c/Users/USER/Desktop/nam/gpt/.env")
api_key = os.getenv('key')
client = OpenAI(api_key=api_key)

def run_conversation(messages, model):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True  # stream을 활성화합니다.
    )
    return response

# 예시 사용
messages = [
    {"role": "system", "content": "You are ChatGPT, a large language model trained by OpenAI."},
    {"role": "user", "content": "안녕하세요."}
]

model = "gpt-4o-2024-05-13"

response = run_conversation(messages, model)

# 스트림 방식으로 응답 받기
# for chunk in response:
#     if chunk.choices[0].delta.content is not None:
#         print(chunk.choices[0].delta.content, end="")
        
for chunk in response:
    if chunk.choices:
        for choice in chunk.choices:
            if choice.delta and choice.delta.content:
                print(choice.delta.content, end="")
        
