import sys, struct, json
sys.path.insert(0, '/mnt/data/catpaw/home/workspace/洛神赋1.1')

# ============================================================
# Part 1: Create a test PPM image file (no libraries)
# ============================================================

def make_test_ppm(path, width=64, height=48):
    """创建一个简单的 PPM P3（ASCII）图片文件。
    描绘：上 1/3 天空（浅蓝），中 1/3 山（深绿）+ 塔（深色），下 1/3 米黄建筑 + 深灰屋顶。
    尺寸：64×48。
    """
    pixels = []
    for y in range(height):
        row = []
        for x in range(width):
            if y < 16:  # 天空
                r, g, b = 180, 210, 240
            elif y < 32:  # 山 + 塔
                if 20 <= x <= 26 and 20 <= y <= 31:  # 塔（深色）
                    r, g, b = 60, 60, 70
                elif 16 <= x <= 30:  # 山（深绿）
                    r, g, b = 40, 100, 50
                else:
                    r, g, b = 180, 210, 240
            else:  # 建筑 + 屋顶
                if y >= 28:  # 屋顶带
                    r, g, b = 80, 80, 90  # 深灰
                else:
                    r, g, b = 220, 200, 160  # 米黄
            row.append((r, g, b))
        pixels.append(row)

    with open(path, 'w') as f:
        f.write('P3\n')
        f.write(f'{width} {height}\n')
        f.write('255\n')
        for row in pixels:
            for (r, g, b) in row:
                f.write(f'{r} {g} {b}\n')
    return path


# ============================================================
# Part 2: Decoder — read raw file, extract pixels (no PIL, no cv2)
# ============================================================

def decode_ppm_p3(path):
    """手工解码 PPM P3（ASCII 格式），返回 [height][width][3] 的字节。"""
    with open(path, 'r') as f:
        lines = f.read().split()

    magic = lines[0]
    assert magic == 'P3', f'Not P3: {magic}'
    w = int(lines[1])
    h = int(lines[2])
    maxval = int(lines[3])

    data = list(map(int, lines[4:]))
    assert len(data) == w * h * 3, f'Wrong data length: {len(data)}, expected {w*h*3}'

    pixels = []
    idx = 0
    for y in range(h):
        row = []
        for x in range(w):
            r, g, b = data[idx], data[idx+1], data[idx+2]
            row.append((r, g, b))
            idx += 3
        pixels.append(row)
    return pixels, w, h


def decode_ppm_p6(path):
    """手工解码 PPM P6（二进制格式），返回 [height][width][3] 的字节。"""
    with open(path, 'rb') as f:
        magic = f.readline().strip()
        assert magic == b'P6', f'Not P6: {magic}'
        # skip comments
        line = f.readline()
        while line.startswith(b'#'):
            line = f.readline()
        w, h = map(int, line.split())
        maxval = int(f.readline().strip())
        raw = f.read()

    assert len(raw) == w * h * 3, f'Wrong raw length: {len(raw)}, expected {w*h*3}'
    pixels = []
    idx = 0
    for y in range(h):
        row = []
        for x in range(w):
            r, g, b = raw[idx], raw[idx+1], raw[idx+2]
            row.append((r, g, b))
            idx += 3
        pixels.append(row)
    return pixels, w, h


def decode_bmp(path):
    """手工解码 BMP（无压缩，24-bit），返回 [height][width][3] 的字节。"""
    with open(path, 'rb') as f:
        data = f.read()

    # BMP header
    assert data[0:2] == b'BM', 'Not BMP'
    pixel_offset = struct.unpack_from('<I', data, 10)[0]
    width = struct.unpack_from('<i', data, 18)[0]
    height = struct.unpack_from('<i', data, 22)[0]
    bpp = struct.unpack_from('<H', data, 28)[0]
    assert bpp == 24, f'Only 24-bit BMP, got {bpp}'
    compression = struct.unpack_from('<I', data, 30)[0]
    assert compression == 0, 'Compressed BMP not supported'

    # BMP rows are bottom-to-top, padded to 4 bytes
    row_size = (width * 3 + 3) & ~3
    pixels = []
    for y in range(abs(height)):
        row_start = pixel_offset + y * row_size
        row = []
        for x in range(width):
            b = data[row_start + x * 3]
            g = data[row_start + x * 3 + 1]
            r = data[row_start + x * 3 + 2]
            row.append((r, g, b))
        pixels.append(row)

    if height > 0:
        pixels.reverse()
    return pixels, width, abs(height)


# ============================================================
# Part 3: 转网格和亮度
# ============================================================

def 转亮度网格(pixels, target_h=48, target_w=64):
    """将任意尺寸的像素图下采样到 target_h × target_w 的亮度网格。

    算法：目标每个格子 = 对应源区域亮度的平均。
    亮度 = 0.299*R + 0.587*G + 0.114*B （ITU-R BT.601）。
    """
    src_h = len(pixels)
    src_w = len(pixels[0])
    grid = []
    for ty in range(target_h):
        row = []
        for tx in range(target_w):
            # 源区域边界
            y0 = int(ty * src_h / target_h)
            y1 = int((ty + 1) * src_h / target_h)
            x0 = int(tx * src_w / target_w)
            x1 = int((tx + 1) * src_w / target_w)
            y1 = max(y1, y0 + 1)
            x1 = max(x1, x0 + 1)

            # 平均亮度
            s = 0.0
            n = 0
            for sy in range(y0, y1):
                for sx in range(x0, x1):
                    r, g, b = pixels[sy][sx]
                    lum = 0.299 * r + 0.587 * g + 0.114 * b
                    s += lum
                    n += 1
            row.append(s / n)
        grid.append(row)
    return grid


def feed_visual_matrix(grid, source_label):
    """将亮度网格喂进视觉矩阵，输出矩阵的统计特征。"""
    import 矩阵.视觉 as visual_m
    A = len(grid)
    B = len(grid[0])

    resampled = _resample_grid(grid, A, B)

        result = visual_m.帧转亮度(resampled)
    mat = result[0] if isinstance(result, tuple) else result
    flat = [v for r in mat for v in r]

    stats = {
        'source': source_label,
        'size': f'{len(mat)}×{len(mat[0])}',
        'mean': round(sum(flat)/len(flat), 4),
        'variance': round(sum((v-sum(flat)/len(flat))**2 for v in flat)/len(flat), 4),
        'min': round(min(flat), 4),
        'max': round(max(flat), 4),
        'range_ratio': round(max(flat) / max(min(flat), 1e-9), 4),
    }
    return mat, stats


def _resample_grid(src, dst_h, dst_w):
    if not src or not src[0]:
        return [[0.0]*dst_w for _ in range(dst_h)]
    src_h = len(src)
    src_w = max(len(r) for r in src)
    out = []
    for i in range(dst_h):
        fi = i * (src_h - 1) / max(dst_h - 1, 1)
        i0 = int(fi); i1 = min(i0+1, src_h-1); wi = fi - i0
        row = []
        for j in range(dst_w):
            fj = j * (src_w - 1) / max(dst_w - 1, 1)
            j0 = int(fj); j1 = min(j0+1, src_w-1); wj = fj - j0
            v = (src[i0][j0]*(1-wi)*(1-wj) + src[i0][j1]*(1-wi)*wj +
                 src[i1][j0]*wi*(1-wj) + src[i1][j1]*wi*wj)
            row.append(v)
        out.append(row)
    return out


# ============================================================
# 主流程
# ============================================================

def main():
    ppm_path = '/mnt/data/catpaw/home/workspace/洛神赋1.1/test_city.ppm'

    # 1. 创建测试图
    make_test_ppm(ppm_path, 64, 48)
    print(f"[1] 创建测试图: {ppm_path}")

    # 2. 手工解码
    pixels, w, h = decode_ppm_p3(ppm_path)
    print(f"[2] 解码: {w}×{h} pixels")

    # 3. 转亮度网格
    grid = 转亮度网格(pixels, 48, 64)
    print(f"[3] 亮度网格: {len(grid)}×{len(grid[0])}")

    # 4. 喂视觉矩阵
    mat, stats = feed_visual_matrix(grid, source_label="test_city_ppm (64×48 → 48×64)")
    print(f"[4] 视觉矩阵统计:")
    for k, v in stats.items():
        print(f"    {k}: {v}")

    # 5. 落痕
    import os, hashlib
    jsonl_path = '/mnt/data/catpaw/home/workspace/洛神赋1.1/luoshenfu_home/traces/' + \
                 'image_decode_' + os.path.basename(ppm_path).replace('.ppm','') + '.jsonl'
    sig = hashlib.md5(open(ppm_path,'rb').read()).hexdigest()[:12]
    record = {
        'type': 'image_decode',
        'source': 'test_city_ppm (generated, no real camera)',
        'file': ppm_path,
        'size_original': f'{w}×{h}',
        'size_resampled': f'{len(grid)}×{len(grid[0])}',
        'stats': stats,
        'hash': sig,
    }
    os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
    with open(jsonl_path, 'a', encoding='utf-8') as fp:
        fp.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f"[5] 落痕: {jsonl_path}")

    return mat


if __name__ == '__main__':
    main()
