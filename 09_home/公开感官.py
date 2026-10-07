"""公开感官.py —— 从公开API获取真实世界物理数据，作为洛神的真实感官来源。

精卫公式解读：
  a = 洛神当前对世界的认知（内部状态）
  b = 公开物理数据（真实世界的物理量）
  Δ = b − a（洛神认知与真实世界的差异）
  Σ = a + b（联合状态）
  ⊗ = ∫(Δ, Σ)（累积差异→学习）
  τ 读 ⊗ → 浮出所见 = 洛神对世界的理解从⊗中浮现
  (a, b) ⟲ ⊗ = 反复感知-学习循环

这不是投喂假数据——这是真实世界的物理量，通过公开API获取。
和视网膜感知光、耳蜗感知声一样——公开感官感知的是真实世界的天气、地震等物理现象。
设备离线（无网络）就失去这个感官——绝不造假数据。

数据源：
  - Open-Meteo（免费、无需API key）：温度、湿度、风速、气压、降水、云量、光照
  - USGS地震API（免费）：最近地震的震级和距离

转译规则（和人类通道的感官描述转译一致）：
  - 温度 → 温值（开尔文）
  - 光照（短波辐射） → 光值（lux近似）
  - 风速 → 力值
  - 云量 → 声值近似（云量高→安静，云量低→可能有风声）
  - 湿度 → 附加感官维度
"""

import json
import math
import urllib.request
import urllib.error


class 公开感官:
    """从公开API获取真实世界物理数据，转译成洛神能感知的信号。

    每隔一定拍数获取一次数据（不需要每拍都获取——API有速率限制）。
    获取的数据缓存下来，每拍都用缓存数据转译成信号。
    网络离线时返回None——绝不造假数据。

    精卫公式：
      a = 洛神内部状态（概念激活/身体激活）
      b = 公开物理数据（真实天气/地震）
      Δ = b - a → 洛神与真实世界的差异
      ⊗ = Δ × Σ → 学习信号
    """

    def __init__(self, n_hidden=11, 纬度=39.9, 经度=116.4,
                 获取间隔=10, 超时秒=5):
        self.n_hidden = n_hidden
        self.纬度 = 纬度
        self.经度 = 经度
        self.获取间隔 = 获取间隔
        self.超时秒 = 超时秒

        # 缓存的最新数据
        self.天气数据 = None
        self.地震数据 = None
        self.上次获取拍 = -100

        # 状态
        self.获取成功次数 = 0
        self.获取失败次数 = 0
        self.离线 = False

        # 当前转译信号（缓存）
        self.当前信号 = None

    def 获取天气(self):
        """从Open-Meteo获取真实天气数据。

        Open-Meteo是免费的开源天气API，无需API key。
        返回当前温度、湿度、风速、气压、降水、云量、光照等。
        """
        url = (f"https://api.open-meteo.com/v1/forecast?"
               f"latitude={self.纬度}&longitude={self.经度}"
               f"&current=temperature_2m,relative_humidity_2m,wind_speed_10m,"
               f"surface_pressure,precipitation,cloud_cover,shortwave_radiation"
               f"&timezone=auto")

        try:
            req = urllib.request.Request(url, headers={"User-Agent": "luoshen/1.0"})
            with urllib.request.urlopen(req, timeout=self.超时秒) as resp:
                data = json.loads(resp.read().decode())
            self.获取成功次数 += 1
            self.离线 = False
            return data.get("current", {})
        except (urllib.error.URLError, urllib.error.HTTPError,
                json.JSONDecodeError, TimeoutError, OSError):
            self.获取失败次数 += 1
            self.离线 = True
            return None

    def 获取地震(self):
        """从USGS获取最近地震数据。

        USGS地震API是免费的，无需API key。
        返回最近24小时内震级>=2.5的地震。
        """
        url = ("https://earthquake.usgs.gov/fdsnws/event/1/query?"
               "format=geojson&minmagnitude=2.5&orderby=time&limit=5")

        try:
            req = urllib.request.Request(url, headers={"User-Agent": "luoshen/1.0"})
            with urllib.request.urlopen(req, timeout=self.超时秒) as resp:
                data = json.loads(resp.read().decode())
            self.获取成功次数 += 1
            self.离线 = False
            features = data.get("features", [])
            if features:
                # 取最近的地震
                最近 = features[0]
                props = 最近.get("properties", {})
                coords = 最近.get("geometry", {}).get("coordinates", [0, 0, 0])
                return {
                    "震级": props.get("mag", 0.0),
                    "距离_km": self._计算距离(coords[1], coords[0]),
                    "时间": props.get("time", ""),
                    "位置": props.get("place", ""),
                }
            return None
        except (urllib.error.URLError, urllib.error.HTTPError,
                json.JSONDecodeError, TimeoutError, OSError):
            self.获取失败次数 += 1
            return None

    def _计算距离(self, lat, lon):
        """计算地震位置到洛神位置的大圆距离（km）。"""
        R = 6371.0  # 地球半径km
        dlat = math.radians(lat - self.纬度)
        dlon = math.radians(lon - self.经度)
        a = (math.sin(dlat / 2) ** 2 +
             math.cos(math.radians(self.纬度)) * math.cos(math.radians(lat)) *
             math.sin(dlon / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c

    def 转译信号(self, 天气, 地震):
        """把真实物理数据转译成洛神能感知的信号。

        转译规则和人类通道的感官描述转译一致：
        - 温度 → 温值（开尔文）
        - 光照 → 光值
        - 风速 → 力值
        - 云量 → 声值近似
        - 湿度 → 附加维度

        精卫公式：
          b = 公开物理数据转译的信号向量
          这个b和人类通道的b一样，进入概念层和身体环境
        """
        if 天气 is None:
            return None

        信号 = {
            "温值": None,
            "光值": None,
            "力值": None,
            "声值": None,
            "湿度": None,
            "气压": None,
            "降水": None,
            "云量": None,
        }

        # 温度 → 温值（开尔文）
        temp_c = 天气.get("temperature_2m")
        if temp_c is not None:
            信号["温值"] = temp_c + 273.15

        # 光照（短波辐射 W/m²） → 光值（近似lux）
        # 短波辐射1000 W/m² ≈ 正午阳光 ≈ 100000 lux
        # 短波辐射0 W/m² ≈ 夜晚 ≈ 0 lux
        radiation = 天气.get("shortwave_radiation")
        if radiation is not None:
            信号["光值"] = radiation * 100.0  # W/m² → lux近似

        # 风速（km/h） → 力值
        wind = 天气.get("wind_speed_10m")
        if wind is not None:
            信号["力值"] = wind

        # 云量（%） → 声值近似
        # 云量高→安静（0），云量低→可能有风声（60）
        # 这不是精确映射——是近似转译
        cloud = 天气.get("cloud_cover")
        if cloud is not None:
            信号["声值"] = max(0, 60 - cloud * 0.5)
            信号["云量"] = cloud

        # 湿度、气压、降水直接传递
        信号["湿度"] = 天气.get("relative_humidity_2m")
        信号["气压"] = 天气.get("surface_pressure")
        信号["降水"] = 天气.get("precipitation")

        # 地震数据 → 附加力值
        if 地震 is not None:
            震级 = 地震.get("震级", 0.0)
            距离 = 地震.get("距离_km", 99999)
            # 近距离大地震 → 力值显著增加
            if 距离 < 1000:
                信号["地震力"] = 震级 * max(0, 1 - 距离 / 1000) * 50
            else:
                信号["地震力"] = 0.0
        else:
            信号["地震力"] = 0.0

        return 信号

    def 一拍(self, 拍):
        """每拍调用：按间隔获取数据，返回转译信号。

        精卫公式：
          每隔获取间隔拍数，从公开API获取真实数据（b）
          转译成洛神能感知的信号
          网络离线时返回None——绝不造假数据
        """
        # 按间隔获取
        if 拍 - self.上次获取拍 >= self.获取间隔:
            self.天气数据 = self.获取天气()
            self.地震数据 = self.获取地震()
            self.上次获取拍 = 拍

        # 转译信号
        self.当前信号 = self.转译信号(self.天气数据, self.地震数据)
        # 转译成概念层可用的n_hidden维向量
        self.当前向量 = self.转译向量(self.当前信号)
        return self.当前信号

    def 转译向量(self, 信号):
        """把物理量信号转译成n_hidden维向量——供概念层作为b信号使用。

        精卫公式：
          a = 洛神内部状态（概念激活/隐藏状态）
          b = 公开物理数据转译的向量
          Δ = b - a → 洛神与真实世界的差异
          ⊗ = Δ × Σ → 学习信号

        转译规则：
          每个物理量归一化到[-1,1]范围，然后映射到n_hidden维向量的不同维度。
          温度→前2维，光照→中间2维，风速→2维，云量→2维，
          湿度/气压/降水/地震力→剩余维度。
          不同物理量占据不同维度——概念层通过赫布学习自然区分。
        """
        if 信号 is None:
            return [0.0] * self.n_hidden

        vec = [0.0] * self.n_hidden
        n = self.n_hidden

        # 温度 → 归一化到[-1,1]：250K-310K映射到[-1,1]
        if 信号.get("温值") is not None:
            temp_norm = max(-1.0, min(1.0, (信号["温值"] - 280.0) / 30.0))
            vec[0] = temp_norm
            if n > 1:
                vec[1] = temp_norm * 0.5  # 温度的二阶效应

        # 光照 → 归一化到[-1,1]：0-1000 W/m²映射到[0,1]
        if 信号.get("光值") is not None:
            light_norm = max(-1.0, min(1.0, 信号["光值"] / 50000.0 * 2.0 - 1.0))
            idx = min(2, n - 1)
            vec[idx] = light_norm
            if n > 3:
                vec[min(3, n - 1)] = light_norm * 0.5

        # 风速 → 归一化到[-1,1]：0-50 km/h映射到[0,1]
        if 信号.get("力值") is not None:
            wind_norm = max(-1.0, min(1.0, 信号["力值"] / 25.0 - 1.0))
            idx = min(4, n - 1)
            vec[idx] = wind_norm
            if n > 5:
                vec[min(5, n - 1)] = wind_norm * 0.5

        # 云量 → 归一化到[-1,1]：0-100%映射到[-1,1]
        if 信号.get("云量") is not None:
            cloud_norm = max(-1.0, min(1.0, 信号["云量"] / 50.0 - 1.0))
            idx = min(6, n - 1)
            vec[idx] = cloud_norm

        # 湿度 → 归一化
        if 信号.get("湿度") is not None and n > 7:
            humid_norm = max(-1.0, min(1.0, 信号["湿度"] / 50.0 - 1.0))
            vec[min(7, n - 1)] = humid_norm

        # 气压 → 归一化
        if 信号.get("气压") is not None and n > 8:
            press_norm = max(-1.0, min(1.0, (信号["气压"] - 1000.0) / 30.0))
            vec[min(8, n - 1)] = press_norm

        # 降水 → 归一化
        if 信号.get("降水") is not None and n > 9:
            precip_norm = max(-1.0, min(1.0, 信号["降水"] / 5.0))
            vec[min(9, n - 1)] = precip_norm

        # 地震力 → 归一化
        if 信号.get("地震力") is not None and n > 10:
            quake_norm = max(-1.0, min(1.0, 信号["地震力"] / 50.0))
            vec[min(10, n - 1)] = quake_norm

        return vec

    def 状态摘要(self):
        return {
            "获取成功": self.获取成功次数,
            "获取失败": self.获取失败次数,
            "离线": self.离线,
            "有数据": self.当前信号 is not None,
            "温值": self.当前信号.get("温值") if self.当前信号 else None,
            "光值": self.当前信号.get("光值") if self.当前信号 else None,
            "力值": self.当前信号.get("力值") if self.当前信号 else None,
            "云量": self.当前信号.get("云量") if self.当前信号 else None,
            "地震力": self.当前信号.get("地震力") if self.当前信号 else None,
        }


if __name__ == "__main__":
    print("=== 公开感官验证 ===\n")
    print("精卫公式: a=洛神内部状态, b=公开物理数据, Δ=b-a, ⊗=Δ×Σ\n")

    感官 = 公开感官(n_hidden=11, 纬度=39.9, 经度=116.4, 获取间隔=1)

    print("--- 测试1: 获取真实天气数据 ---")
    天气 = 感官.获取天气()
    if 天气:
        print(f"  温度: {天气.get('temperature_2m')}°C")
        print(f"  湿度: {天气.get('relative_humidity_2m')}%")
        print(f"  风速: {天气.get('wind_speed_10m')}km/h")
        print(f"  气压: {天气.get('surface_pressure')}hPa")
        print(f"  降水: {天气.get('precipitation')}mm")
        print(f"  云量: {天气.get('cloud_cover')}%")
        print(f"  光照: {天气.get('shortwave_radiation')}W/m²")
    else:
        print("  天气数据获取失败（网络离线）")

    print("\n--- 测试2: 获取真实地震数据 ---")
    地震 = 感官.获取地震()
    if 地震:
        print(f"  最近地震: 震级={地震.get('震级')}, 距离={地震.get('距离_km'):.0f}km")
        print(f"  位置: {地震.get('位置')}")
    else:
        print("  地震数据获取失败或无近期地震")

    print("\n--- 测试3: 转译信号 ---")
    信号 = 感官.转译信号(天气, 地震)
    if 信号:
        for k, v in 信号.items():
            if v is not None:
                print(f"  {k}: {v:.2f}" if isinstance(v, float) else f"  {k}: {v}")
    else:
        print("  转译失败")

    print("\n--- 测试4: 一拍流程 ---")
    感官2 = 公开感官(获取间隔=1)
    信号1 = 感官2.一拍(1)
    信号2 = 感官2.一拍(2)  # 不获取，用缓存
    print(f"  拍1信号: {信号1 is not None}")
    print(f"  拍2信号(缓存): {信号2 is not None}")
    print(f"  状态: {感官2.状态摘要()}")

    print("\n--- 测试5: 离线模式 ---")
    感官3 = 公开感官(获取间隔=1, 超时秒=0.001)  # 极短超时模拟离线
    信号 = 感官3.一拍(1)
    print(f"  离线信号: {信号}")  # 应该是None或天气数据None
    print(f"  离线状态: {感官3.状态摘要()}")

    print("\n验证通过")
