import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio


def create_base_fig(df_node, df_face):
    # lobe 列が存在すると仮定（ない場合は Label_name → lobe の辞書で map してください）
    lobe_categories = df_node["Label_name"].astype("category")
    df_node["lobe_id"] = lobe_categories.cat.codes  # 例：Frontal→0, Temporal→1, ...

    # カラーマップ定義（カテゴリ順）
    lobe_colors = {
        # Lobe base
        "Frontal":   "#e4f2f7",  # light blue (元: #b4d9eb)
        "Parietal":  "#fefeda",  # light yellow (元: #fdfb97)
        "Temporal":  "#e4eddb",  # light green (元: #b7cf9d)
        "Occipital": "#fad5dd",   # faded light pink
        "Limbic/Medial": "#eeeeee",

        # Modality base
        'Auditory':            '#e6f0ff',  # 淡いブルー
        'Language':            '#fff0cc',  # 淡いイエロー
        'Emotion':             '#ffe6ec',  # 淡いピンク
        'Executive':           '#d9f7e6',  # 淡いミントグリーン
        'Gustatory':           '#fff6d9',  # 淡い黄緑がかったクリーム
        'Motor':               '#e0e0ff',  # 薄いパープル（青寄り）
        'Somatosensory':       '#e6ffe6',  # 淡い黄緑（自然系）
        'Visual':              '#f0e6ff',  # 淡いラベンダー（青紫）

        # 混合モダリティ：構成色の中間（できるだけ補色寄せでコントラスト）
        'Auditory; Language':     '#f0f0e6',  # 青 × 黄 → 淡グレー寄り
        'Executive; Emotion':     '#f2eae6',  # ミント × ピンク → コーラルグレイ
        'Gustatory; Emotion':     '#fae6ef',  # 黄緑 × ピンク → 淡桃
        'Visual; Emotion':        '#f9e6f5',  # 紫 × ピンク → ピンク寄りラベンダー
        'Visual; Language':       '#f2e6ff'   # 紫 × 黄 → 青みパステル
    }
    # カテゴリ順に並べて colorscale 用に変換
    categories = lobe_categories.cat.categories
    discrete_colorscale = [
        [i / (len(categories) - 1), lobe_colors.get(cat, "gray")]
        for i, cat in enumerate(categories)
    ]

    # --- 左右統合メッシュ（連続で描画） ---

    # 頂点データ
    v_all = df_node[["x", "y", "z"]].values
    intensity_all = df_node["lobe_id"].values

    # 面データ（整数インデックスで、df_node に対応していると仮定）
    f_all = df_face[["i", "j", "k"]].values  # v1, v2, v3 は面の頂点列


    # 頂点と面（numpy arrayである前提）
    v_L = df_node[df_node["LR"] == "L"].values
    f_L = df_face[df_face["LR"] == "L"].values

    v_R = df_node[df_node["LR"] == "R"].values
    f_R = df_face[df_face["LR"] == "R"].values


    # === 左メッシュ (L, 青) ===
    mesh3d_L = go.Mesh3d(
        x=v_L[:, 0], y=v_L[:, 1], z=v_L[:, 2],
        i=f_L[:, 0], j=f_L[:, 1], k=f_L[:, 2],
        intensity=v_L[:, 6],            # 数値 → カラースケール
        colorscale=discrete_colorscale,
        opacity=1,
        name='Left',
        flatshading=False,  # 影付き（必要に応じて調整）
        showscale=False
    )

    # === 右メッシュ (R, 赤) ===
    mesh3d_R = go.Mesh3d(
        x=v_R[:, 0], y=v_R[:, 1], z=v_R[:, 2],
        i=f_R[:, 0], j=f_R[:, 1], k=f_R[:, 2],
        intensity=v_R[:, 6],            # 数値 → カラースケール
        colorscale=discrete_colorscale,
        opacity=1,
        name='Right',
        flatshading=False,
        showscale=False
    )

    fig = go.Figure(data=[mesh3d_L, mesh3d_R])
    fig.update_layout(
        title='Brain Mesh Colored by Lobe (No Mesh Cutting)',
        scene=dict(xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False)),
        margin=dict(l=0, r=0, b=0, t=0),
        showlegend=False
    )
    # --- Label_name に実際に含まれるカテゴリだけ抽出 ---
    used_labels = df_node["Label_name"].astype(str).unique()

    # --- レジェンド用のダミー Trace を必要なものだけ追加 ---
    legend_traces = []
    for label in used_labels:
        color = lobe_colors.get(label, "gray")
        legend_traces.append(
            go.Scatter3d(
                x=[None], y=[None], z=[None],
                mode='markers',
                marker=dict(size=6, color=color),
                name=label,
                showlegend=True
            )
        )

    legend_fig = go.Figure(data=legend_traces)
    legend_fig.update_layout(
        title='Lobe Legend',
        showlegend=True,
        margin=dict(l=0, r=0, b=0, t=30),
        height=400,
        width=300,
        scene=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False),
        legend=dict(
            x=0,  # ← ここを調整して左寄せにする
            y=0.5,
            xanchor='left',
            yanchor='middle',
            traceorder='normal',
            font=dict(size=10),
            bgcolor='rgba(0,0,0,0)'
        )
    )

    return fig, legend_fig


def add_channel_mesh(fig, nodes_df, faces_df, channel, side, opacity=0.75, show_colorbar=False):
    vn = nodes_df[(nodes_df['Channel']==channel) & (nodes_df['LR']==side)].sort_values('v_local')
    ff = faces_df[(faces_df['Channel']==channel) & (faces_df['LR']==side)]
    if vn.empty or ff.empty:
        return

    # dist の逆数（ゼロ割防止・外れ値ケア）
    d = vn['dist'].to_numpy(float)
    invd = 1.0 / (d + 1e-6)
    # 無限/NaN を除去（あれば最大有限値に置換）
    if not np.isfinite(invd).all():
        finite = invd[np.isfinite(invd)]
        repl = finite.max() if finite.size else 1.0
        invd = np.where(np.isfinite(invd), invd, repl)

    fig.add_trace(go.Mesh3d(
        x=vn['x'], y=vn['y'], z=vn['z'],
        i=ff['i'].astype(int), j=ff['j'].astype(int), k=ff['k'].astype(int),
        intensity=invd,
        colorscale='Viridis',        # viridis(option="C")
        cmin=float(invd.min()), cmax=float(invd.max()),
        opacity=opacity,
        name=f'{channel}-{side}',
        showscale=show_colorbar
    ))



def add_all_mni_points(fig, mni, size=4, color='black', label=False,text_pos='top center'):
    if label:
        fig.add_trace(go.Scatter3d(
            x=mni['x'], y=mni['y'], z=mni['z'],
            mode='markers+text',
            text=mni['Channel'],
            textposition=text_pos,
            textfont=dict(size=15),
            marker=dict(size=size, color=color, opacity=0.9),
            name='MNI points',
            showlegend=False
        ))
    else:
        fig.add_trace(go.Scatter3d(
            x=mni['x'], y=mni['y'], z=mni['z'],
            mode='markers+text',
            marker=dict(size=size, color=color, opacity=0.9),
            name='MNI points',
            showlegend=False
        ))

def plot_meshes(df_ref,
                ref_map,
                df_node,
                df_face,
                mni,
                channel_node,
                channel_face,
                color_label = "lobe"):
    # 前処理: ref_map のカラム名が揃っているか確認
    # "harvard_oxford" でマージする
    df_merged = df_ref.merge(ref_map[["harvard_oxford", color_label]],
                            left_on="Label_name", right_on="harvard_oxford",
                            how="left")

    # 不要な列（harvard_oxford）を削除するなら
    df_merged.drop(columns=["harvard_oxford"], inplace=True)


    # --- Step 1: 座標データを NumPy に変換 ---
    target_coords = df_node[['x', 'y', 'z']].values
    ref_coords = df_ref[['x', 'y', 'z']].values

    # --- Step 2: 参照点群にKDTreeを構築 ---
    tree = cKDTree(ref_coords)

    # --- Step 3: 最近傍点を検索（戻り値: 各点に対する (距離, インデックス)） ---
    dists, idxs = tree.query(target_coords, k=1)

    # --- Step 4: 最近傍の index を元に ID を付与（または座標を追加） ---
    df_node['Label_name'] = df_merged[color_label].values[idxs]
    df_node['Label_ID'] = df_ref["Label_ID"].values[idxs]

    fig, legend_fig = create_base_fig(df_node, df_face)  # 修正済み関数を使用
    fig1 = go.Figure(fig)

    for channel in channel_node["Channel"].unique():
        for side in ['L', 'R']:
            add_channel_mesh(fig, channel_node, channel_face, channel, side, opacity=1)
    fig.update_layout(scene=dict(aspectmode='data'))
    fig2 = go.Figure(fig)

    # 使い方：脳全体メッシュやチャネルメッシュを追加した後に呼ぶ
    add_all_mni_points(fig, mni, size=4, color='black')
    fig.update_layout(scene=dict(aspectmode='data'))
    fig3 = go.Figure(fig)

    add_all_mni_points(fig, mni, size=4, color='black', label=True)
    fig.update_layout(scene=dict(aspectmode='data'))
    fig4 = go.Figure(fig)


    # 1行5列に変更
    row = make_subplots(rows=1, cols=5, specs=[[{'type':'scene'}]*5],
                        subplot_titles=(f"{color_label} level", "Channel distance", "Channel points", "Channel labels", ""),
                        column_widths=[0.25, 0.25, 0.25, 0.25, 0],
                        horizontal_spacing=0.01)


    # 既存のトレースを各セルへコピー
    # 各figのdataをサブプロットへ
    for tr in fig1.data: row.add_trace(tr, row=1, col=1)
    for tr in fig2.data: row.add_trace(tr, row=1, col=2)
    for tr in fig3.data: row.add_trace(tr, row=1, col=3)
    for tr in fig4.data: row.add_trace(tr, row=1, col=4)
    for tr in legend_fig.data: row.add_trace(tr, row=1, col=5)


    # 軸非表示・アスペクト固定
    # 軸非表示・アスペクト固定（1〜5列）
    for c in (1,2,3,4,5):
        row.update_scenes(dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False,
                            aspectmode="data"), row=1, col=c)


    # カメラ統一
    if 'scene' in fig1.layout and 'camera' in fig1.layout.scene:
        cam = fig1.layout.scene.camera
        for s in range(1, 6):
            row.update_layout({f'scene{s}_camera': cam})

    return row


df_ref = pd.read_csv('data/1_HO_NMI.csv')
ref_map = pd.read_csv('data/0_Harvard_Oxford_Brodmann_Lobe.csv')
df_node = pd.read_csv('data/2_LR_NMI_node.csv')
df_face = pd.read_csv('data/2_LR_NMI_face.csv')
mni = pd.read_csv("data/0_Okamoto_M_2004_table2_MNI_coordinates.csv")
channel_node = pd.read_csv("data/3_LR_NMI_channel_node.csv")
channel_face = pd.read_csv("data/3_LR_NMI_channel_face.csv")


row = plot_meshes(df_ref, ref_map, df_node, df_face, mni, channel_node, channel_face, color_label = "lobe")
row.update_layout(height=450, width=1500, margin=dict(l=0,r=0,t=40,b=0), showlegend=True)
pio.write_html(row, "data/4_result_lobe.html", include_plotlyjs="inline", full_html=True)


row = plot_meshes(df_ref, ref_map, df_node, df_face, mni, channel_node, channel_face, color_label = "MotorSensory")
row.update_layout(height=450, width=1500, margin=dict(l=0,r=0,t=40,b=0), showlegend=True)
pio.write_html(row, "data/4_result_motor_sensory.html", include_plotlyjs="inline", full_html=True)