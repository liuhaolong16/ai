# -*- coding: utf-8 -*-
"""用 GitHub API git/blobs 下载 NASA 电池 .mat 文件（urllib + 重试）"""
import urllib.request, json, base64, os, time

repo = 'changyeon99/Battery-Data-Set'
# --- repository-relative paths (added for the public deposit; see README.md) ---
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA = os.path.join(_REPO, "data")
_RESULTS = os.path.join(_REPO, "results")
_FIGURES = os.path.join(_REPO, "figures")
DATA = _DATA
os.makedirs(DATA, exist_ok=True)

files = {
    'B0005.mat': '8ecf6c78caf437a36fec60cf02470510b46e9162',
    'B0006.mat': '8867cf00c30a521848d83b2d31be3e162710e7ad',
    'B0007.mat': 'd6289aa0c6b3126d98cfe7fdecfaec4a78d5eaad',
    'B0018.mat': '13ef3ffda5be655edc2ed08a52073aedf73b548b',
}

for name, sha in files.items():
    out = os.path.join(DATA, name)
    if os.path.exists(out) and os.path.getsize(out) > 1000000:
        print(f"SKIP {name} (已存在 {os.path.getsize(out)} bytes)")
        continue
    url = f"https://api.github.com/repos/{repo}/git/blobs/{sha}"
    ok = False
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'dsh', 'Accept': 'application/vnd.github+json'})
            with urllib.request.urlopen(req, timeout=900) as resp:
                data = json.loads(resp.read().decode('utf-8'))
            content = data['content'].replace('\n', '')
            raw = base64.b64decode(content)
            with open(out, 'wb') as f:
                f.write(raw)
            print(f"OK {name} {len(raw)} bytes (attempt {attempt+1})")
            ok = True
            break
        except Exception as e:
            print(f"FAIL {name} attempt {attempt+1}: {e}")
            time.sleep(8)
    if not ok:
        print(f"GIVE UP {name}")
print("PY DOWNLOAD DONE")
