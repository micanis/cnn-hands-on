---
layout: two-cols
---

::title::
環境構築ファイルの更新

::left::

<Transform :scale="0.8">

```python
from google.colab import drive
drive.mount("/content/drive")


###
# 授業開始時にこのセルを実行すること
###

import os
import shutil

# --- パスの設定 ---
git_url = "https://github.com/micanis/cnn-hands-on.git"
volatile_repo = "/content/cnn-hands-on-repo"
drive_dir = "/content/drive/MyDrive/cnn-hands-on"

# クローン
!rm -rf {volatile_repo}
!git clone {git_url} {volatile_repo}

# 2. フォルダ作成
os.makedirs(drive_dir, exist_ok=True)
os.makedirs(os.path.join(drive_dir, "notebooks"), exist_ok=True)

print("フォルダ構成をGoogle Driveに同期中")

# 3. ファイルコピー
src_notebooks = os.path.join(volatile_repo, "workshop", "notebooks")
dst_notebooks = os.path.join(drive_dir, "notebooks")

if os.path.exists(src_notebooks):
    for file_name in os.listdir(src_notebooks):
        if file_name.endswith(".ipynb"):
            src_file = os.path.join(src_notebooks, file_name)
            dst_file = os.path.join(dst_notebooks, file_name)
            
            # Driveに存在しないノートブックだけを追加
            if not os.path.exists(dst_file):
                shutil.copy(src_file, dst_file)
                print(f"新しいノートブックを追加しました: notebooks/{file_name}")

src_utils = os.path.join(volatile_repo, "workshop", "utils")
dst_utils = os.path.join(drive_dir, "utils")

if os.path.exists(src_utils):
    if os.path.exists(dst_utils):
        shutil.rmtree(dst_utils) # 古いシステムを削除
    shutil.copytree(src_utils, dst_utils)

req_src = os.path.join(volatile_repo, "workshop", "requirements.txt")
req_dst = os.path.join(drive_dir, "requirements.txt")
if os.path.exists(req_src):
    shutil.copy(req_src, req_dst)

src_models = os.path.join(volatile_repo, "workshop", "models")
dst_models = os.path.join(drive_dir, "models")
if os.path.exists(src_models):
    shutil.copytree(src_models, dst_models, dirs_exist_ok=True)

print("Ready!!")
# 作業ディレクトリをDrive内に移動
os.chdir(drive_dir)
```

</Transform>

::right::
詳しくは授業中の指示に従ってください

1. Google Driveからcnn-hands-onフォルダを開き```colab_setup.ipynb```を開く
2. 全てのコードを削除する
3. 左のコードをコピペして新規セルを作成して貼り付ける
