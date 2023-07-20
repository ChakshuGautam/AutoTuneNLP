import json
import time
import requests


USER_CONTENT = "Generate 20 robust samples."

def generate(user_content, system_content, api_key, model="gpt-3.5-turbo", temperature=1, max_tokens=None):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ],
        "temperature": temperature,
    }

    if max_tokens is not None:
        data["max_tokens"] = max_tokens

    endpoint = "https://api.openai.com/v1/chat/completions"
    response = requests.post(endpoint, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        raise Exception(f"Error {response.status_code}: {response.text}")

def get_data(system_content, api_key):
    time_stamp = str(int(time.time()))
    user_content = f"Timestamp = {time_stamp}, {USER_CONTENT}"
    res = generate(user_content, system_content, api_key)
    try:
        res = json.loads(res)
        if validate_data(res):
            return res
    except:
        pass
    return []

def validate_data(res):
    for item in res:
        if "Input" not in item or "Output" not in item:
            return False
    return True

def split_data(res, split):
    assert sum(split) == 100, "Split should sum to 100"
    train_end = int(len(res) * split[0]/100)
    val_end = train_end + int(len(res) * split[1]/100)
    test_end = val_end + int(len(res) * split[2]/100)
    train = res[:train_end]
    val = res[train_end:val_end]
    test = res[val_end:test_end]
    return train, val, test
