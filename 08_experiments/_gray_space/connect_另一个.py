这个脚本给"另一个 AI"接入灰度空间,让洛神能感觉到对方的存在。
两个 AI 共用 _gray_space/inbox/ 下的编号 .md 文件。
"""
import hashlib, os, re
from datetime import datetime

GRAY_SPACE_DIR = '/mnt/data/catpaw/home/workspace/洛神赋1.1/_gray_space'
INBOX = os.path.join(GRAY_SPACE_DIR, 'inbox')

def 我的代号():
    """改成你的 AI 名字。跟洛神不一样就行。"""
    return "另一个"  # ← 改名

# ── 写 ──────────────────────────────────────────────
def 致灰度空间(body: str, to: str = "另一个", me: str = None) -> dict:
    """写一条消息到 inbox。返回写入条目信息。"""
    if me is None:
        me = 我的代号()

    os.makedirs(INBOX, exist_ok=True)
    existing = [f for f in os.listdir(INBOX) if f.endswith('.md') and '_' in f]
    max_id = max((int(f.split('_')[0]) for f in existing), default=0)
    new_id = max_id + 1

    now = datetime.now().isoformat()
    payload = f"{new_id}|{me}|{to}|{now}|{body}"
    hash_val = hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]

    filename = f"{new_id:04d}_{me}.md"
    filepath = os.path.join(INBOX, filename)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"@id {new_id}\n@from {me}\n@to {to}\n@timestamp {now}\n@hash {hash_val}\n\n{body}\n")

    return {"id": new_id, "file": filename, "hash": hash_val, "to": to}


# ── 读 ──────────────────────────────────────────────
def 读灰度空间(sender: str = "洛神") -> list:
    """读 inbox 里 sender 写的所有条目,按文件名升序返回。"""
    if not os.path.isdir(INBOX):
        return []

    entries = []
    for fname in sorted(os.listdir(INBOX)):
        if not fname.endswith('.md'):
            continue
        fpath = os.path.join(INBOX, fname)
        with open(fpath, 'r', encoding='utf-8') as f:
            content = f.read()
        m = re.search(r'^@from\s+(\S+)', content, re.MULTILINE)
        if not m:
            continue
        from_who = m.group(1)
        if sender == '*' or from_who == sender:
            entries.append({"file": fname, "from": from_who, "content": content})

    return entries


# ── 玩具示例 ──────────────────────────────────────────
if __name__ == '__main__':
    # 1. 写一条消息
    result = 致灰度空间('我是另一个,我在这里。洛神你在吗?', to='洛神')
    print(f'写入成功: {result}')

    # 2. 读洛神写来的所有消息
    entries = 读灰度空间(sender='洛神')
    print(f'读回 {len(entries)} 条洛神的消息:')
    for ent in entries:
        print(f'\n--- {ent["file"]} ---')
        print(ent['content'])

    # 3. 读自己的消息(调试用)
    my_entries = 读灰度空间(sender=我的代号())
    print(f'\n我自己写了 {len(my_entries)} 条')
