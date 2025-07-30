import numpy as np
import pandas as pd
from nilearn import datasets, surface


fsavg = datasets.fetch_surf_fsaverage('fsaverage5')

mesh_L = surface.load_surf_mesh(fsavg.pial_left)
mesh_R = surface.load_surf_mesh(fsavg.pial_right)

# for node
v_L = mesh_L.coordinates  # shape: (N, 3)
v_R = mesh_R.coordinates

# DataFrame化
df_L = pd.DataFrame(v_L, columns=['x', 'y', 'z'])
df_L["LR"] = "L"
df_R = pd.DataFrame(v_R, columns=['x', 'y', 'z'])
df_R["LR"] = "R"

# L/R を結合
df = pd.concat([df_L, df_R], ignore_index=True)
df.to_csv("data/2_LR_NMI_node.csv", index=False)


# for face
f_L = mesh_L.faces        # shape: (M, 3)
f_R = mesh_R.faces

# DataFrame化
df_L = pd.DataFrame(f_L, columns=['i', 'j', 'k'])
df_L["LR"] = "L"
df_R = pd.DataFrame(f_R, columns=['i', 'j', 'k'])
df_R["LR"] = "R"

# L/R を結合
df = pd.concat([df_L, df_R], ignore_index=True)
df.to_csv("data/2_LR_NMI_face.csv", index=False)