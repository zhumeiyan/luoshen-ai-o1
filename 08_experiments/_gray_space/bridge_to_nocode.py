#!/usr/bin/env python3
"""
灰空间 → NoCode桥：把洛神最新身体状态转成JSON写到本地文件，
由NoCode预览面板热重载取读。
猫爪每跑一轮引擎后执行一次，把洛神身体映射到3D场景的可视化数据。

用法：python3 bridge_to_nocode.py [--chat-id CHAT_ID]
（chat-id暂未内联调用，仅生成本地JSON预览）
"""
import json, os, sys, time

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRAY = os.path.join(WORKSPACE, "_gray_space")
OUTBOX = os.path.join(GRAY, "outbox")

def 读最新快照():
    """读outbox里最新一条洛神快照，返回字段dict"""
    if not os.path.isdir(OUTBOX):
        return None
    files = sorted([f for f in os.listdir(OUTBOX) if f.endswith(".md") and f.split("_")[0].isdigit()])
    if not files:
        return None
    latest = files[-1]
    with open(os.path.join(OUTBOX, latest)) as f:
        content = f.read()
    data = {}
    in_body = False
    for line in content.split("\n"):
        if in_body:
            if "=" in line:
                k, v = line.split("=", 1)
                try:
                    data[k] = float(v) if "." in v else int(v)
                except ValueError:
                    data[k] = v
        elif line == "":
            in_body = True
    data["_file"] = latest
    data["_time"] = time.time()
    return data

def 默认状态():
    return {
        "tick": 0,
        "hunger": 0.0,
        "存货": 400,
        "情绪色价": 0.5,
        "筋树数": 0,
        "总激活": 0.0,
        "最大激活": 0.0,
        "节点亮": 0,
        "前5亮": []
    }

if __name__ == "__main__":
    data = 读最新快照()
    if not data:
        数据 = 默认状态()
        数据["_note"] = "无快照, 使用默认"
    else:
        数据 = {**默认状态(), **数据}
    out = os.path.join(WORKSPACE, "_gray_space", "bridge_state.json")
    with open(out, "w") as f:
        json.dump(数据, f, ensure_ascii=False, indent=2)
    print("写入", out)
    for k in ["tick", "筋树数", "情绪色价", "节点亮"]:
        print(f"  {k}: {数据.get(k, '?')}")
