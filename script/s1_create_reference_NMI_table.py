import numpy as np
import pandas as pd
from nilearn import datasets, image

harvard_oxford = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr0-2mm')


img = image.load_img(harvard_oxford.maps)
data = img.get_fdata()
affine = img.affine
labels = harvard_oxford.labels  # index 0 = Background

label_ids = []
label_names = []

for i, label_name in enumerate(labels):
    if label_name == 'Background':
        continue
    label_ids.append(i)
    label_names.append(label_name)


rows = []  # 空のリストを用意

for i, j, k in np.ndindex(data.shape):
    if data[i, j, k] == 0:
        continue  # Backgroundをスキップ

    label_index = data[i, j, k]
    label_name = label_names[int(label_index) - 1]

    # ボクセル空間の重心を計算
    voxel_coord = np.array([i, j, k])

    # MNI座標に変換
    com_mni = np.dot(affine, list(voxel_coord) + [1])[:3]

    rows.append({
        "Label_ID": int(label_index),
        "Label_name": label_name,
        "x": com_mni[0],
        "y": com_mni[1],
        "z": com_mni[2]
    })

df = pd.DataFrame(rows)
df.sort_values("Label_ID", inplace=True)

df.to_csv('data/1_HO_NMI.csv', index=False)