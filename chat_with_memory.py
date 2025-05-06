import json
import os
from openai import OpenAI
import datetime

MEMORY_FILE = os.path.join(os.path.dirname(__file__), 'memories.json')
LOG_FILE = os.path.join(os.path.dirname(__file__), 'chat_with_memory.log')

client = OpenAI(
    api_key = "your-api-key",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

def log_message(title, content, level="info"):
    lines = content.strip().splitlines()
    if len(lines) > 1:
        content_fmt = "\n" + "\n".join(["    " + line for line in lines])
    else:
        content_fmt = " " + lines[0] if lines else ""
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"[{level.upper()}][{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {title}{content_fmt}\n")

def load_memories():
    if not os.path.exists(MEMORY_FILE):
        return []
    with open(MEMORY_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_memories(memories):
    with open(MEMORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(memories, f, ensure_ascii=False, indent=2)

def get_related_memory_ids(memories, user_prompt, messages):
    # 第一次调用LLM，发送完整记忆内容，返回相关记忆编号
    prompt = f"""
你是一个记忆检索助手。以下是历史记忆：
{json.dumps(memories, ensure_ascii=False, indent=2)}

用户输入：{user_prompt}

历史对话：{json.dumps(messages, ensure_ascii=False, indent=2)}

请以JSON数组形式返回与用户输入和上下文相关的记忆编号，例如：[1,3,5]，只返回编号即可。
"""
    log_message("[Request] get_related_memory_ids prompt", prompt)
    response = client.chat.completions.create(
        model="deepseek-v3",
        messages=[{"role": "user", "content": prompt}],
        stream=False
    )
    log_message("[Response] get_related_memory_ids", response.choices[0].message.content)
    ids = []
    try:
        ids = json.loads(response.choices[0].message.content)
    except Exception:
        pass
    return ids

def get_memory_by_ids(memories, ids):
    return [m for m in memories if m.get('id') in ids]

def update_memories_with_llm(memories, related_memories, user_prompt, messages):
    # 第三次调用LLM，返回记忆变更
    prompt = f"""
你是一个记忆管理助手。以下是与本次对话相关的记忆：
{json.dumps(related_memories, ensure_ascii=False, indent=2)}

用户输入：{user_prompt}

历史对话：{json.dumps(messages, ensure_ascii=False, indent=2)}

请根据上下文判断是否有新的**长期事实、习惯、偏好、关系、身份、计划等**需要记住。
请始终优先考虑保存有用的信息。如果用户透露了新事实，而记忆中没有，请新增记忆。
请以JSON数组形式返回本次对话后需要新增、修改或删除的记忆。
- 新增记忆：编号为-1，内容为新增内容。
- 修改记忆：编号为原编号，内容为修改后的内容。
- 删除记忆：编号为要删除的id，内容为null或"delete"。
例如：
[
  {{"id": -1, "content": "新记忆内容"}},
  {{"id": 2, "content": "修改后的记忆内容"}},
  {{"id": 3, "content": null}}
]
如果没有变更，返回空数组[]。
"""
    log_message("[Request] update_memories_with_llm prompt", prompt)
    response = client.chat.completions.create(
        model="deepseek-v3",
        messages=[{"role": "system", "content": "请判断是否有记忆需要新增、修改或删除。"}, {"role": "user", "content": prompt}],
        stream=False
    )
    log_message("[Response] update_memories_with_llm", response.choices[0].message.content)
    try:
        changes = json.loads(response.choices[0].message.content)
    except Exception:
        changes = []
    # 更新memories
    max_id = max([m.get('id', 0) for m in memories], default=0)
    to_delete = set()
    for change in changes:
        if change['id'] == -1:
            max_id += 1
            memories.append({'id': max_id, 'content': change['content']})
        elif change.get('content') is None or (isinstance(change.get('content'), str) and change.get('content').lower() == 'delete'):
            to_delete.add(change['id'])
        else:
            for m in memories:
                if m.get('id') == change['id']:
                    m['content'] = change['content']
    if to_delete:
        memories = [m for m in memories if m.get('id') not in to_delete]
    return memories

def chat_with_memory():
    memories = load_memories()
    messages = []
    while True:
        user_prompt = input("你：")
        if user_prompt.strip().lower() in ["exit", "quit", "q"]:
            break
        # 第一次调用LLM，获取相关记忆编号（发送完整记忆内容）
        recent_messages = messages[-10:]  # 只保留最近5轮（10条）上下文
        related_ids = get_related_memory_ids(memories, user_prompt, recent_messages)
        related_memories = [m for m in memories if m.get('id') in related_ids]
        # 第二次调用LLM，生成最终回复（发送完整记忆内容）
        prompt = f"""
以下是与本次对话相关的记忆：
{json.dumps(related_memories, ensure_ascii=False, indent=2)}

用户输入：{user_prompt}

历史对话：{json.dumps(recent_messages, ensure_ascii=False, indent=2)}
请根据记忆和上下文，回复用户。
"""
        log_message("[Request] final reply prompt", prompt)
        response = client.chat.completions.create(
            model="deepseek-v3",
            messages=[{"role": "user", "content": prompt}],
            stream=False
        )
        log_message("[Response] final reply", response.choices[0].message.content)
        reply = response.choices[0].message.content
        print("AI：" + reply)
        messages.append({"role": "user", "content": user_prompt})
        messages.append({"role": "assistant", "content": reply})
        # 第三次调用LLM，获取记忆变更
        memories = update_memories_with_llm(memories, related_memories, user_prompt, recent_messages + [{"role": "user", "content": user_prompt}, {"role": "assistant", "content": reply}])
        save_memories(memories)

if __name__ == "__main__":
    chat_with_memory()
