import streamlit as st

import json
import pandas as pd
import numpy as np
import os
import re
import copy
import time

from openai import OpenAI

st.set_page_config(layout="wide")

gonggo = st.radio("공고 선택", ['지역난방공사', '수자원공사'], horizontal=True, label_visibility='collapsed')

st.write(os.getcwd())
if gonggo == '지역난방공사':
    file_path = os.getcwd() + '/datas/jinanbang'
else:
    file_path = os.getcwd() + '/datas/sujawon_jeonmunjik'

with open(file_path + '/df_to_json_with_vertor.json', 'r') as f:
    json_data = json.load(f)
    
json_df = pd.DataFrame(json_data)

for_assistant_without_vector = copy.deepcopy(json_data)
for row in for_assistant_without_vector:
    del row['embedded_vector']
    
with open(file_path + "/system_prompt.txt", 'r') as f:
    system_prompt = "".join([r for r in f])
with open(file_path + "/first_assistant.txt", 'r') as f:
    first_assistant = "".join([r for r in f])
first_assistant = f"기준질문-답변 데이터 : {for_assistant_without_vector}" + first_assistant

model = "gpt-4o-2024-05-13"

##############################################################################
##### local
try:
    from dotenv import load_dotenv
    load_dotenv("/mnt/c/Users/USER/Desktop/nam/gpt/.env")
    api_key = os.getenv('key')

##### streamlit
except:
    api_key = st.secrets["api_key"]
##############################################################################
client = OpenAI(api_key=api_key)

def stream_data(m_c):
    for word in m_c.split(" "):
        yield word + " "
        time.sleep(0.05)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_embedding(text, model):
    return client.embeddings.create(input = [text], model=model).data[0].embedding

def normalize_l2(x):
    x = np.array(x)
    if x.ndim == 1:
        norm = np.linalg.norm(x)
        if norm == 0:
            return x
        return x / norm
    else:
        norm = np.linalg.norm(x, 2, axis=1, keepdims=True)
        return np.where(norm == 0, x, x / norm)
    
def search_docs(j_df, user_query):
    embedding = get_embedding(
        user_query,
        model="text-embedding-3-small"
    )
    small_emb = normalize_l2(embedding[:512])
    j_df["similarities"] = j_df['embedded_vector'].apply(lambda x: cosine_similarity(x, small_emb))
    res = j_df.sort_values("similarities", ascending=False)
    res = res.loc[res['similarities'] > 0.45][1:6]
    return res
       
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
          
if f'messages_{gonggo}' not in st.session_state:
    st.session_state[f'messages_{gonggo}'] = []
    st.session_state[f'messages_{gonggo}'].append({"role": "system", "content": system_prompt})
    st.session_state[f'messages_{gonggo}'].append({"role": "assistant", "content": first_assistant})
    
col1, col2 = st.columns(2)
with col1:
    st.header("FAQ - GPT.ver")
    chat_1 = st.container(height=610)
    input_1 = st.container(height=73)
    with chat_1:
        button_col1, button_col2 = st.columns([0.85,0.15])
        with button_col2:
            reset_button = st.button("대화 초기화")
            if reset_button:
                st.session_state[f'messages_{gonggo}'] = st.session_state[f'messages_{gonggo}'][:2]
        if len(st.session_state[f'messages_{gonggo}']) > 3:
            write_messages = st.session_state[f'messages_{gonggo}'][2:]
            for mdx, message in enumerate(write_messages):
                if type(message) == dict:
                    if message["role"] in ['user', 'assistant']:
                        with st.chat_message(message["role"]):
                            st.write(message["content"])
                    if message["role"] in ['tool']:
                        with st.chat_message(message["role"]):
                            st.write(message["content"])
                            tool_result = message["content"]
            
    with input_1:
        user_input = st.chat_input("질문을 입력하세요.")
        if user_input:
            st.session_state[f'messages_{gonggo}'].append({"role": "user", "content": user_input})
            
    if user_input:
        with chat_1:
            with st.chat_message('user'):
                st.write(user_input)
            messages = st.session_state[f'messages_{gonggo}'][:2] + [st.session_state[f'messages_{gonggo}'][-1]]
            response_message = run_conversation(messages, model)
            answer_role = response_message.choices[0].message.role
            answer_content = response_message.choices[0].message.content
            st.session_state[f'messages_{gonggo}'].append({"role": answer_role, "content": answer_content})
            with st.chat_message(answer_role):
                st.write_stream(stream_data(answer_content))
                if "해당 질문에 대한 적절한 답변을 찾을 수 없습니다." not in answer_content:
                    no1_question = answer_content.split('"')[1].split("]")[-1].strip()
                    res = search_docs(json_df, no1_question)
                    if len(res) > 0:
                        with st.popover("유사질문 확인하기"):
                            for rdx, row in res.iterrows():
                                label = f"**[{rdx+1}] [{row['구분']}] {row['질문']}**"
                                with st.expander(label, expanded=False):
                                    st.divider()
                                    st.write(row['답변'])

with col2:
    st.header("FAQ - BOARD.ver")
    with st.container(height=700):
        for rdx, row in json_df.iterrows():
            label = f"**[{rdx+1}] [{row['구분']}] {row['질문']}**"
            with st.expander(label, expanded=False):
                st.divider()
                st.write(row['답변'])