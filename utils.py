import json
import os
import streamlit as st
from configs.config import USER_CONFIG_PATH


def save_data(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)


# 从 JSON 文件加载数据
def load_data(data_path):
    with open(data_path, 'r') as f:
        data = json.load(f)
    return data


def find_all_files(path, target_file_types):
    res = []
    if os.path.exists(path):
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.split('.')[-1] in target_file_types:
                    res.append(os.path.join(root, file))
    return res


def init_session_state(state_path=USER_CONFIG_PATH):
    session_state = load_data(state_path)
    for k, v in session_state.items():
        if k not in st.session_state:
            st.session_state[k] = v
