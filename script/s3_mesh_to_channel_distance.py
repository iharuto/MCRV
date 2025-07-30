import re
import numpy as np
import pandas as pd
import matplotlib.tri as mtri

# ---------- ルール：チャネルからLRグループを決める ----------
def lr_groups_from_channel(channel: str):
    """
    z/Z を含む: ['L','R'] の2グループを返す（別々に再構成）
    末尾数字が偶数: ['R']
    末尾数字が奇数: ['L']
    それ以外: ['L','R']（フォールバック）
    """
    s = str(channel)
    if re.search(r'[zZ]', s):
        return ['L', 'R']
    m = re.search(r'(\d+)$', s)
    if m:
        n = int(m.group(1))
        return ['R'] if (n % 2 == 0) else ['L']
    return ['L', 'R']

# ---------- 近傍ノード選抜（TopK） ----------
def nearest_topk_nodes(point_xyz, df_node, allowed_lr={'L','R'}, k=100):
    mask_lr = df_node['LR'].isin(list(allowed_lr))
    nodes_lr = df_node[mask_lr].copy()
    if len(nodes_lr) == 0:
        # 返す形式を合わせる
        out = pd.DataFrame(columns=list(df_node.columns) + ['dist','node_idx'])
        return out

    coords = nodes_lr[['x','y','z']].to_numpy(dtype=float)
    p = np.asarray(point_xyz, dtype=float)[None, :]
    dists = np.linalg.norm(coords - p, axis=1)
    order = np.argsort(dists)[:min(k, len(nodes_lr))]
    selected = nodes_lr.iloc[order].copy()
    selected['dist'] = dists[order]
    selected['node_idx'] = selected.index  # 元 df_node 上のIDを保持
    return selected.reset_index(drop=True)  # 0..N-1 に再インデックス（facesはこの番号を参照）

# ---------- 既存faceの抽出＆再インデックス ----------
def subset_and_reindex_faces(df_face, selected_nodes, side: str, require_face_lr=True):
    """
    side: 'L' または 'R'
    require_face_lr=True: face の LR も side に一致させる
    """
    if selected_nodes is None or len(selected_nodes) < 3:
        return pd.DataFrame(columns=['i','j','k'])

    # 元ノードID(=df_node index) → 新インデックス(=selected_nodesの行番号 0..N-1)
    orig_ids_in_order = selected_nodes['node_idx'].tolist()
    id_map = {orig: new for new, orig in enumerate(orig_ids_in_order)}
    have = set(orig_ids_in_order)

    faces = df_face
    if require_face_lr and 'LR' in df_face.columns:
        faces = faces[faces['LR'] == side]

    if len(faces) == 0:
        return pd.DataFrame(columns=['i','j','k'])

    tri = faces[['i','j','k']].to_numpy(dtype=int)
    keep_mask = np.isin(tri, list(have)).all(axis=1)
    faces_keep = faces.loc[keep_mask, ['i','j','k']].copy()
    if len(faces_keep) == 0:
        return pd.DataFrame(columns=['i','j','k'])

    faces_keep['i'] = faces_keep['i'].map(id_map)
    faces_keep['j'] = faces_keep['j'].map(id_map)
    faces_keep['k'] = faces_keep['k'].map(id_map)
    return faces_keep

# ---------- 2Dドロネーでのface再構成 ----------
def triangulate_faces_from_points(selected_nodes,
                                  plane_sigma=2.0,
                                  edge_factor=2.0,
                                  min_triangles=1):
    """
    selected_nodes: 0..N-1 に再インデックス済みで ['x','y','z'] を持つ DataFrame
    plane_sigma: 平面からの外れ点除去のしきい（標準偏差×）
    edge_factor: 長過ぎる辺の除去しきい（最近傍距離の中央値×）
    """
    if selected_nodes is None or len(selected_nodes) < 3:
        return pd.DataFrame(columns=['i','j','k']), {'reason': 'too_few_points'}

    P = selected_nodes[['x','y','z']].to_numpy(float)
    N = len(P)

    # PCA/SVDで局所平面推定 → 2Dへ射影
    C = P.mean(axis=0)
    X = P - C
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    normal = Vt[2, :]
    basis2 = Vt[:2, :].T
    UV = X @ basis2

    # 平面から外れた点を除去（任意）
    keep_mask = np.ones(N, dtype=bool)
    if plane_sigma is not None:
        signed_h = X @ normal
        thr = plane_sigma * (np.std(signed_h) + 1e-9)
        keep_mask = np.abs(signed_h) <= thr

    if not keep_mask.all() and keep_mask.sum() >= 3:
        P2 = P[keep_mask]
        UV2 = UV[keep_mask]
        idx_global = np.nonzero(keep_mask)[0]
    else:
        P2 = P
        UV2 = UV
        idx_global = np.arange(N)

    if len(P2) < 3:
        return pd.DataFrame(columns=['i','j','k']), {'reason': 'too_few_points_after_outlier'}

    # 2D Delaunay（三角分割）
    tri2d = mtri.Triangulation(UV2[:,0], UV2[:,1])
    faces = tri2d.triangles.copy()
    if faces.size == 0:
        return pd.DataFrame(columns=['i','j','k']), {'reason': 'delaunay_empty'}

    # 長過ぎる辺の除去（品質フィルタ）
    D = np.linalg.norm(P2[:, None, :] - P2[None, :, :], axis=2)
    np.fill_diagonal(D, np.inf)
    nn = D.min(axis=1)
    edge_thresh = edge_factor * np.median(nn)

    a = np.linalg.norm(P2[faces[:,1]] - P2[faces[:,0]], axis=1)
    b = np.linalg.norm(P2[faces[:,2]] - P2[faces[:,1]], axis=1)
    c = np.linalg.norm(P2[faces[:,0]] - P2[faces[:,2]], axis=1)
    good = (a <= edge_thresh) & (b <= edge_thresh) & (c <= edge_thresh)

    faces = faces[good]
    if len(faces) < min_triangles:
        faces = tri2d.triangles.copy()  # 最低枚数確保のため緩和

    if len(faces) == 0:
        return pd.DataFrame(columns=['i','j','k']), {'reason': 'all_triangles_filtered'}

    # 法線向きの一貫化（陰影安定用）
    n_tri = np.cross(P2[faces[:,1]] - P2[faces[:,0]],
                     P2[faces[:,2]] - P2[faces[:,0]])
    s = n_tri @ normal
    flip = s < 0
    if np.any(flip):
        faces[flip] = faces[flip][:, [0,2,1]]

    # 局所番号 → selected_nodes の 0..N-1 へ戻す
    faces_global = idx_global[faces]
    faces_df = pd.DataFrame(faces_global, columns=['i','j','k'])
    info = dict(n_points=len(P2), edge_thresh=float(edge_thresh), n_faces=len(faces_df))
    return faces_df, info

# ---------- メイン：各チャネル×LRグループごとにverticesとfacesを返す ----------
def build_channel_mesh_groups(mni, df_node, df_face,
                              topk=100,
                              prefer='existing_then_triangulate',
                              triangulate_kwargs=None,
                              require_face_lr=True):
    """
    Returns:
      mesh_groups: dict keyed by (Channel, LR) =>
        {
          'channel': str,
          'LR': 'L' or 'R',
          'vertices': DataFrame [x,y,z,LR,Label_name,Label_ID,lobe_id, dist, node_idx],
          'faces': DataFrame [i,j,k],
          'source_faces': 'existing' or 'reconstructed',
          'n_vertices': int,
          'n_faces': int
        }
      summary: DataFrame 集計
    """
    if triangulate_kwargs is None:
        triangulate_kwargs = dict(plane_sigma=2.0, edge_factor=2.0, min_triangles=1)

    mesh_groups = {}
    rows = []

    for _, r in mni.iterrows():
        ch = r['Channel']
        p = (r['x'], r['y'], r['z'])
        sides = lr_groups_from_channel(ch)
        if len(sides) == 2:
            topk_ = int(topk / 2)  # 2グループなら半分ずつ
        else:
            topk_ = topk

        for side in sides:
            # 1) ノード選抜
            df_node_ = df_node[df_node["LR"] == side].copy()
            df_face_ = df_face[df_face["LR"] == side].copy()
            df_node_.index = range(len(df_node_))  # インデックスを0..N-1に再設定
            df_face_.index = range(len(df_face_))  # インデックスを0..N-1に再設定

            sel_nodes = nearest_topk_nodes(p, df_node_, allowed_lr={side}, k=topk_)

            # 2) face の取得方針
            faces_df = pd.DataFrame(columns=['i','j','k'])
            source_faces = None

            if prefer in ('existing', 'existing_then_triangulate'):
                faces_df = subset_and_reindex_faces(df_face_, sel_nodes, side=side, require_face_lr=require_face_lr)
                if len(faces_df) > 0:
                    source_faces = 'existing'

            if (len(faces_df) == 0) and (prefer in ('triangulate', 'existing_then_triangulate')):
                faces_df, info = triangulate_faces_from_points(sel_nodes, **triangulate_kwargs)
                if len(faces_df) > 0:
                    source_faces = 'reconstructed'

            mesh_groups[(ch, side)] = dict(
                channel=ch,
                LR=side,
                vertices=sel_nodes,   # 0..N-1 に並んだ頂点（node_idx: 元df_nodeのindex）
                faces=faces_df,       # i,j,k は上記 0..N-1 を参照
                source_faces=source_faces,
                n_vertices=len(sel_nodes),
                n_faces=len(faces_df)
            )

            rows.append({
                'Channel': ch,
                'LR': side,
                'n_vertices': len(sel_nodes),
                'n_faces': len(faces_df),
                'source_faces': source_faces
            })

    summary = pd.DataFrame(rows)
    return mesh_groups, summary


df_node = pd.read_csv('data/2_LR_NMI_node.csv')
df_face = pd.read_csv('data/2_LR_NMI_face.csv')
mni = pd.read_csv("data/0_Okamoto_M_2004_table2_MNI_coordinates.csv")

# 例：Top100で選抜し、まず既存faceを試し、無ければ再構成
mesh_groups, summary = build_channel_mesh_groups(
    mni, df_node, df_face,
    topk=100,
    prefer='existing_then_triangulate',   # 'existing' / 'triangulate' も可
    triangulate_kwargs=dict(plane_sigma=2.0, edge_factor=2.0, min_triangles=1),
    require_face_lr=True                  # 既存face利用時、faceのLRも厳密一致
)


nodes_rows, faces_rows = [], []

for (ch, side), g in mesh_groups.items():
    if not g or g.get('vertices') is None or len(g['vertices']) == 0:
        continue

    # vertices を追加
    v = g['vertices'].copy().reset_index(drop=True)
    v['v_local'] = np.arange(len(v), dtype=int)  # 念のため明示列
    v['Channel'] = ch
    v['LR'] = side
    nodes_rows.append(v)

    # faces を追加（空ならスキップ）
    f = g.get('faces')
    if f is not None and len(f) > 0:
        f2 = f[['i','j','k']].astype(int).copy()
        f2['Channel'] = ch
        f2['LR'] = side
        faces_rows.append(f2)

# まとめ
nodes_df = (pd.concat(nodes_rows, ignore_index=True)
            if nodes_rows else pd.DataFrame(columns=['x','y','z','Channel','LR','v_local']))
faces_df = (pd.concat(faces_rows, ignore_index=True)
            if faces_rows else pd.DataFrame(columns=['Channel','LR','i','j','k']))


nodes_df.to_csv("data/3_LR_NMI_channel_node.csv", index=False)
faces_df.to_csv("data/3_LR_NMI_channel_face.csv", index=False)
