import streamlit as st

import json
import pandas as pd
import numpy as np
import os
import re
import copy

from openai import OpenAI
# from dotenv import load_dotenv

st.set_page_config(layout="wide")

df = pd.read_excel('./한난_23하 공채_FAQ_최종_230816_수정.xlsx')
df.dropna(axis=0, inplace=True)
df = df[1:].reset_index(drop=True)
df.columns = ['순번', '구분', '질문', '답변']

##### local
# load_dotenv("/mnt/c/Users/USER/Desktop/nam/gpt/.env")
# api_key = os.getenv('key')

##### streamlit
api_key = st.secrets["api_key"]

client = OpenAI(api_key=api_key)

# model = "gpt-3.5-turbo-16k-0613"

def get_current_weather(location, unit="fahrenheit"):
    """Get the current weather in a given location"""
    if "tokyo" in location.lower():
        return json.dumps({"location": "Tokyo", "temperature": "10", "unit": unit})
    elif "san francisco" in location.lower():
        return json.dumps({"location": "San Francisco", "temperature": "72", "unit": unit})
    elif "paris" in location.lower():
        return json.dumps({"location": "Paris", "temperature": "22", "unit": unit})
    else:
        return json.dumps({"location": location, "temperature": "unknown"})

def run_conversation(messages, model):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        }
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice="auto", 
    )
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls
    if tool_calls:
        available_functions = {
            "get_current_weather": get_current_weather,
        } 
        messages.append(response_message)
        for tool_call in tool_calls:
            print("함수 호출 :", tool_call)
            print("-"*100)
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)
            function_response = function_to_call(**function_args)
            print("호출 결과 :", function_response)
            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                }
            )
        second_response = client.chat.completions.create(
            model=model,
            messages=messages,
        )
        return second_response
    return response

###############################################################################################################

prompt = '''당신은 채용시스템의 FAQ기능을 대신합니다. 
            사용자의 질문에 올바른 답변을 합니다. 
            기준질문과 답변이 1:1로 매칭되어있는 아래 정보를 토대로 가장 맥락이 비슷한 답변을 내놓습니다.
            채용관련한 질문이 아니라면 "채용FAQ시스템입니다. 채용에 관련된 내용으로만 질문해주세요."라고 답변합니다.'''
first_assistant = f"기준질문-답변 데이터 : {df.to_json(orient='records',force_ascii=False)}"
if 'messages_GPT-4o' not in st.session_state:
    st.session_state['messages_GPT-4o'] = []
    st.session_state['messages_GPT-4o'].append({"role": "system", "content": prompt})
    st.session_state['messages_GPT-4o'].append({"role": "assistant", "content": first_assistant})
    
if 'messages_GPT-3.5' not in st.session_state:
    st.session_state['messages_GPT-3.5'] = []
    st.session_state['messages_GPT-3.5'].append({"role": "system", "content": prompt})
    st.session_state['messages_GPT-3.5'].append({"role": "assistant", "content": first_assistant})

if 'user_input' not in st.session_state:
    st.session_state['user_input'] = ''

col1, col2 = st.columns(2)
with col1:
    st.header("FAQ - GPT.ver")
    radio = st.radio("모델 선택", ['GPT-4o', 'GPT-3.5'], horizontal=True, label_visibility='collapsed')
    if radio == 'GPT-4o':
        model = "gpt-4o-2024-05-13"
    else:
        model = "gpt-3.5-turbo-16k-0613"

    chat_1 = st.container(height=568)
    with st.container(height=73):
        user_input = st.chat_input("질문을 입력하세요.")
        st.session_state['user_input'] = user_input
        if user_input:
            st.session_state[f'messages_{radio}'].append({"role": "user", "content": st.session_state['user_input']})
            messages = st.session_state[f'messages_{radio}']
            response_message = run_conversation(messages, model)
            answer_role = response_message.choices[0].message.role
            answer_content = response_message.choices[0].message.content
            st.session_state[f'messages_{radio}'].append({"role": answer_role, "content": answer_content})
    with chat_1:
        if len(st.session_state[f'messages_{radio}']) > 3:
            for message in st.session_state[f'messages_{radio}'][2:]:
                if type(message) == dict:
                    if message["role"] in ['user', 'assistant']:
                        with st.chat_message(message["role"]):
                            st.write(message["content"])
                    if message["role"] in ['tool']:
                        with st.chat_message(message["role"]):
                            st.write(message["content"])
                        tool_result = message["content"]
with col2:
    st.header("FAQ - BORAD.ver")
    with st.container(height=700):
        for rdx, row in df.iterrows():
            label = f"**[{rdx+1}] [{row['구분']}] {row['질문']}**"
            with st.expander(label, expanded=False):
                st.divider()
                st.write(row['답변'])
# try:
#     messages.append({"role": response_message.role, "content": response_message.content})
# except:
#     func_response = response_message.choices[0].message
#     messages.append({"role": func_response.role, "content": func_response.content})
    
# print("@"*100)
# print()
# for m in messages[2:]:
#     if type(m) == dict:
#         print(m['role'].upper())
#         print("-"*30)
#         print(m['content'])
#         print()
# print("@"*100)    
