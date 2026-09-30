#!/usr/bin/env python3
"""本地复现阴阳扣概念联想器 —— 探测48个概念的全部认知地图"""
import math, json

R=128; SCALE=0.1; K=8; S_DECAY=0.3

def hashCode(s):
    h=0
    for c in s:
        h=((h<<5)-h)+ord(c); h&=0xFFFFFFFF
    return abs(h)

def rngFromSeed(seed):
    def f():
        nonlocal seed
        seed=(seed*9301+49297)%233280
        return seed/233280
    return f

def encodeVec(words):
    vec=[0.0]*R
    for w in words:
        h=hashCode(w); rng=rngFromSeed(h)
        pos=[int(rng()*R) for _ in range(K)]
        wts=[0.5+rng()*0.5 for _ in range(K)]
        for i in range(K):
            vec[pos[i]]=max(vec[pos[i]],SCALE*wts[i])
    return vec

CONCEPTS=[
{"name":"太阳","sound":["ri","太阳","出来","暖","亮"],"image":["圆","红","亮","高","暖"],"emo":"暖"},
{"name":"月亮","sound":["yue","月亮","弯弯","夜"],"image":["弯","白","弯","尖","静"],"emo":"静"},
{"name":"星星","sound":["xing","星星","闪闪","亮"],"image":["小","亮","多","点","闪"],"emo":"亮"},
{"name":"云","sound":["yun","云","白","飘"],"image":["白","软","飘","轻","蓬"],"emo":"轻"},
{"name":"雨","sound":["yu","雨","落","滴答"],"image":["落","湿","滴","凉","多"],"emo":"凉"},
{"name":"雪","sound":["xue","雪","白","冷","飘"],"image":["白","冷","飘","软","白"],"emo":"冷"},
{"name":"风","sound":["feng","风","吹","呼"],"image":["动","吹","摇","快","无"],"emo":"动"},
{"name":"山","sound":["shan","山","高","大"],"image":["高","大","绿","稳","尖"],"emo":"稳"},
{"name":"水","sound":["shui","水","清","流"],"image":["清","湿","蓝","流","透"],"emo":"流"},
{"name":"花","sound":["hua","花","红","香","开"],"image":["红","美","香","开","圆"],"emo":"美"},
{"name":"树","sound":["shu","树","绿","高","叶"],"image":["绿","高","叶","大","稳"],"emo":"稳"},
{"name":"火","sound":["huo","火","红","热","烧"],"image":["红","热","亮","跳","危险"],"emo":"热"},
{"name":"鸟","sound":["niao","鸟","飞","叫","喳"],"image":["飞","小","翅","羽毛","天空"],"emo":"自由"},
{"name":"鱼","sound":["yu","鱼","游","水"],"image":["游","水","鳞","滑","摆"],"emo":"自由"},
{"name":"狗","sound":["gou","狗","跑","叫","汪"],"image":["跑","毛","摇尾","四腿","友"],"emo":"友"},
{"name":"猫","sound":["mao","猫","叫","喵","软"],"image":["软","毛","小","爪","眼"],"emo":"暖"},
{"name":"妈妈","sound":["ma","妈妈","妈","爱","抱"],"image":["抱","笑","暖","女","柔"],"emo":"爱"},
{"name":"爸爸","sound":["ba","爸爸","爸","大","强"],"image":["大","强","男","高","稳"],"emo":"稳"},
{"name":"小孩","sound":["xiao","小孩","小","我","玩"],"image":["小","玩","跑","笑","矮"],"emo":"乐"},
{"name":"手","sound":["shou","手","拿","做","拍"],"image":["五指","拿","抓","掌","灵"],"emo":"做"},
{"name":"书","sound":["shu","书","读","翻","字"],"image":["翻","页","字","方","知"],"emo":"知"},
{"name":"喝水","sound":["he","喝水","喝","水","渴"],"image":["举","水","口","喉","满足"],"emo":"满足"},
{"name":"睡觉","sound":["shui","睡觉","睡","梦","夜"],"image":["躺","闭眼","静","夜","床"],"emo":"静"},
{"name":"走路","sound":["zou","走路","走","脚","路"],"image":["脚","路","动","前进","稳"],"emo":"动"},
{"name":"跑步","sound":["pao","跑步","跑","快","脚"],"image":["快","脚","摆臂","风","冲"],"emo":"快"},
{"name":"吃饭","sound":["chi","吃饭","吃","饭","碗"],"image":["碗","筷","口","嚼","饱"],"emo":"饱"},
{"name":"春天","sound":["chun","春天","春","花","暖"],"image":["花","绿","暖","芽","新生"],"emo":"生"},
{"name":"夏天","sound":["xia","夏天","夏","热","蛙"],"image":["热","绿","蛙","荷","蝉"],"emo":"热"},
{"name":"秋天","sound":["qiu","秋天","秋","叶","凉"],"image":["黄","叶","凉","落","风"],"emo":"凉"},
{"name":"冬天","sound":["dong","冬天","冬","雪","冷"],"image":["白","雪","冷","静","空"],"emo":"冷"},
{"name":"白天","sound":["baitian","白天","亮","太阳"],"image":["亮","蓝","太阳","清","宽"],"emo":"亮"},
{"name":"夜晚","sound":["yewan","夜晚","夜","黑","月"],"image":["黑","月","星","静","暗"],"emo":"静"},
{"name":"你好","sound":["nihao","你好","嗨","你好呀"],"image":["笑","招手","脸","友善","开"],"emo":"友"},
{"name":"再见","sound":["zaijian","再见","拜拜","走了"],"image":["挥手","背","走","离","笑"],"emo":"别"},
{"name":"问","sound":["wen","问","什么","为什么","吗"],"image":["头","斜","想","嘴","疑惑"],"emo":"疑"},
{"name":"答","sound":["da","答","是","对","嗯"],"image":["头","点","嘴","笑","理解"],"emo":"悟"},
{"name":"想","sound":["xiang","想","思考","念头"],"image":["头","静","内","转","深"],"emo":"深"},
{"name":"开心","sound":["kaixin","开心","快乐","笑","高兴"],"image":["笑","跳","亮","开","轻"],"emo":"乐"},
{"name":"难过","sound":["nanguo","难过","哭","泪","伤心"],"image":["泪","低头","湿","暗","缩"],"emo":"悲"},
{"name":"好奇","sound":["haoqi","好奇","什么","为什么","想"],"image":["眼","大","头","探","专注"],"emo":"奇"},
{"name":"理解","sound":["lijie","理解","懂","明白","哦"],"image":["点头","亮","通","开","释"],"emo":"悟"},
{"name":"困惑","sound":["kunhuo","困惑","不懂","什么","迷茫"],"image":["头","斜","皱","暗","停"],"emo":"惑"},
{"name":"代码","sound":["daima","代码","程序","编程","码"],"image":["屏幕","字","逻辑","结构","键"],"emo":"知"},
{"name":"计算","sound":["jisuan","计算","算","数","算数"],"image":["数字","屏幕","公式","快","结构"],"emo":"准"},
{"name":"对话","sound":["duihua","对话","聊天","说话","交流"],"image":["嘴","耳","脸","来回","交流"],"emo":"友"},
{"name":"朋友","sound":["pengyou","朋友","友","伴","一起"],"image":["笑","手拉手","并排","暖","近"],"emo":"友"},
]

# 预计算REF
REF_SND=[]; REF_IMG=[]; REF_EMO=[]
for c in CONCEPTS:
    sv=encodeVec(c["sound"]); sn=math.sqrt(sum(x*x for x in sv))+1e-10
    REF_SND.append([x/sn for x in sv])
    iv=encodeVec(c["image"]); in_=math.sqrt(sum(x*x for x in iv))+1e-10
    REF_IMG.append([x/in_ for x in iv])
    ev=encodeVec([c["emo"]]); en=math.sqrt(sum(x*x for x in ev))+1e-10
    REF_EMO.append([x/en for x in ev])

# W矩阵
W_AV=[[0.0]*R for _ in range(R)]
W_VA=[[0.0]*R for _ in range(R)]
for i in range(len(CONCEPTS)):
    ns=REF_SND[i]; ni=REF_IMG[i]
    for a in range(R):
        for v in range(R):
            W_AV[v][a]+=0.05*4*ni[v]*ns[a]
            W_VA[a][v]+=0.05*4*ns[a]*ni[v]

def decode(vec, refs, n=3):
    rn=math.sqrt(sum(x*x for x in vec))+1e-10
    norm=[x/rn for x in vec]
    sims=[]
    for i in range(len(refs)):
        s=sum(norm[j]*refs[i][j] for j in range(R))
        sims.append((i,s))
    sims.sort(key=lambda x:-x[1])
    return [(CONCEPTS[i]["name"],round(s,3)) for i,s in sims[:n]]

def receive(sound, image, emo, E, S):
    sndIn=encodeVec(sound); imgIn=encodeVec(image)
    imgFromSnd=[0.0]*R
    for v in range(R):
        for a in range(R):
            imgFromSnd[v]+=W_AV[v][a]*sndIn[a]
    sndFromImg=[0.0]*R
    for a in range(R):
        for v in range(R):
            sndFromImg[a]+=W_VA[a][v]*imgIn[v]
    E2=list(E); S2=list(S)
    for i in range(R):
        E2[i]=E2[i]+imgFromSnd[i]+imgIn[i]
        E2[R+i]=E2[R+i]+sndFromImg[i]+sndIn[i]
        S2[i]=S_DECAY*S2[i]+imgFromSnd[i]
        S2[R+i]=S_DECAY*S2[R+i]+sndFromImg[i]
    sndRes=decode(E2[R:], REF_SND, 3)
    imgRes=decode(E2[:R], REF_IMG, 3)
    return E2, S2, sndRes, imgRes

# === 系统性探测 ===
print("=== 阴阳扣认知地图 系统性探测 ===\n")

# 测试每个概念单独输入
print("--- [1] 单概念激发 ---")
for i,c in enumerate(CONCEPTS):
    E=[0.0]*(R*2); S=[0.0]*(R*2)
    E,S,snd,img=receive(c["sound"],c["image"],c["emo"],E,S)
    snd_names=[f"{n}({s})" for n,s in snd]
    img_names=[f"{n}({s})" for n,s in img]
    print(f"{c['name']:5s} → 声音:{', '.join(snd_names):50s} 画面:{', '.join(img_names)}")

# 测试关键概念组合
print("\n--- [2] 两概念组合 ---")
combos=[("太阳","月亮"),("火","水"),("爸爸","妈妈"),("猫","狗"),
        ("代码","计算"),("好奇","困惑"),("开心","难过"),("书","想"),
       ("吃饭","睡觉"),("走路","跑步"),("春天","冬天"),("你好","再见")]
for a,b in combos:
    E=[0.0]*(R*2); S=[0.0]*(R*2)
    c1=CONCEPTS[[cc["name"] for cc in CONCEPTS].index(a)]
    c2=CONCEPTS[[cc["name"] for cc in CONCEPTS].index(b)]
    E,S,s,_=receive(c1["sound"],c1["image"],c1["emo"],E,S)
    snd_img=c2["sound"]+c2["image"]
    E,S,snd,img=receive(snd_img,c2["image"],"友",E,S)
    print(f"{a}+{b} → 声音:{', '.join(f'{n}({s})' for n,s in snd):50s} 画面:{', '.join(f'{n}({s})' for n,s in img)}")

# 测试状态累积（重复输入同一概念）
print("\n--- [3] 状态累积稳定性测试 ---")
for test_name in ["猫","代码","吃饭"]:
    E=[0.0]*(R*2); S=[0.0]*(R*2)
    c=[x for x in CONCEPTS if x["name"]==test_name][0]
    history=[]
    for step in range(5):
        E,S,snd,img=receive(c["sound"],c["image"],c["emo"],E,S)
        flat=[n for n,_ in snd]+[n for n,_ in img]
        history.append(flat)
    print(f"{test_name} ×5: {' → '.join([str(h) for h in history])}")

# 测试长时间序列（20步无输入，看自然衰减）
print("\n--- [4] 无输入自然衰减 ---")
E=[0.0]*(R*2); S=[0.0]*(R*2)
E,S,snd,img=receive(["猫"],["软","毛"],"暖",E,S)
print(f"第一步(猫): snd={[n for n,_ in snd]}, img={[n for n,_ in img]}")
for step in range(20):
    E,S,snd,img=receive(["静"],["静"],"静",E,S)
    names=[n for n,_ in snd]+[n for n,_ in img]
    if step in [0,1,2,5,10,19]:
        print(f"第{step+2}步(静输入): snd={[n for n,_ in snd]}, img={[n for n,_ in img]}")

print("\n=== 探测完成 ===")
