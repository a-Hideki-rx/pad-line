import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import japanize_matplotlib  # noqa: F401
import matplotlib.lines as mlines
from matplotlib.collections import LineCollection

# =========================
# Page
# =========================
st.set_page_config(layout="wide")
st.title('Pad-Line')
st.markdown(
    '患者ごとの治療ライン上に、**副作用の発現期間（開始マーカー＋期間線）**と'
    '**介入アクション（休薬・減量などの点イベント）**を統合して可視化します。'
    '\n\n'
    '✅ **swimmer / swimlane** 切替、フィルタ、並び替え、同日重なり回避（自動段数化）を搭載。'
)

uploaded_file = st.file_uploader("データファイル（CSV）をアップロードしてください", type='csv')

# =========================
# Utilities
# =========================
def get_color_by_grade(grade):
    """Grade->色（あなたのルール踏襲）"""
    if pd.isna(grade):
        return '#808080'  # グレー
    g = int(grade)
    if g >= 3:
        return '#FF4B4B'  # 濃い赤
    elif g == 2:
        return '#FFA07A'  # オレンジ
    elif g == 1:
        return '#FFE4B5'  # 薄い黄色
    elif g == 0:
        return '#FFFFFF'  # 白
    else:
        return '#808080'


def marker_for_ae(name: str) -> str:
    """AE名（イベント種別の文字列）からマーカー形状へ"""
    s = str(name)
    if '末梢神経障害' in s:
        return '*'
    elif '皮膚障害' in s:
        return '^'
    elif '味覚異常' in s:
        return 'D'
    else:
        return 'o'


def lane_for_event(event_type: str) -> str:
    """単一患者スイムレーン用：イベント種別からカテゴリレーン名へ"""
    s = str(event_type)
    if s in ['休薬', '減量']:
        return '介入（休薬/減量）'
    if '末梢神経障害' in s:
        return '末梢神経障害'
    if '皮膚障害' in s:
        return '皮膚障害'
    if '味覚異常' in s:
        return '味覚異常'
    return 'その他'


def is_ongoing_value(v) -> bool:
    """継続中フラグの表記揺れ吸収（空欄以外を継続扱い）"""
    if v is None:
        return False
    s = str(v).strip()
    return (s != '' and s.lower() != 'nan')


def marker_scale_for_count(cnt: int) -> float:
    """同日イベントが多い時にマーカーを少し縮める（破綻防止）"""
    if cnt <= 3:
        return 1.0
    elif cnt <= 5:
        return 0.88
    else:
        return 0.78


def stable_patients_order(df_in: pd.DataFrame, sort_mode: str, treatment_map: dict) -> list:
    pts = df_in['患者ID'].dropna().unique().tolist()
    if sort_mode == "患者ID（昇順）":
        return sorted(pts)
    if sort_mode == "治療期間（長い順）":
        return sorted(pts, key=lambda p: (treatment_map.get(p, 0), p), reverse=True)
    if sort_mode == "治療期間（短い順）":
        return sorted(pts, key=lambda p: (treatment_map.get(p, 0), p))
    if sort_mode == "イベント数（多い順）":
        counts = df_in.groupby('患者ID').size().to_dict()
        return sorted(pts, key=lambda p: (counts.get(p, 0), p), reverse=True)
    if sort_mode == "最大Grade（高い順）":
        maxg = df_in.groupby('患者ID')['Grade'].max().fillna(-1).to_dict()
        return sorted(pts, key=lambda p: (maxg.get(p, -1), p), reverse=True)
    return sorted(pts)


# ===== 薬剤師向けプリセット（難しいパラメータを隠す）=====
def preset_to_layout(clarity_level: int):
    """
    clarity_level: 0(俯瞰重視)〜100(詳細重視)
    直感的なUIから内部パラメータへ変換する
    """
    expand_threshold = int(np.clip(round(4 - 2.0 * (clarity_level / 100)), 2, 6))
    max_lanes = int(np.clip(round(3 + 5.0 * (clarity_level / 100)), 1, 10))
    gap = float(np.clip(0.5 + 0.6 * (clarity_level / 100), 0.1, 2.0))
    inner_step = float(np.clip(0.03 + 0.03 * (clarity_level / 100), 0.0, 0.12))
    inner_max = float(np.clip(0.06 + 0.05 * (clarity_level / 100), 0.0, 0.20))
    return expand_threshold, max_lanes, gap, inner_step, inner_max


def plot_single_patient_swimlanes(
    df_pt: pd.DataFrame,
    patient_id,
    treatment_duration: float,
    line_width: int = 3,
    base_facecolor: str = "#F8F9FA",
    x_margin: int = 10,
    show_legends: bool = True,
    show_ongoing_arrow: bool = True,
    show_intervention_text: bool = False,
):
    """
    単一患者のスイムレーン描画（カテゴリ別）
    - 横軸：投与開始からの日数
    - 縦軸：カテゴリ（末梢/皮膚/味覚/その他/介入）
    - AE：期間線 + 開始点（色=Grade, 形=AEカテゴリ）
    - 介入：点イベント（休薬/減量）
    """
    d = df_pt.copy()
    d['lane'] = d['イベント種別'].astype(str).apply(lane_for_event)
    d['_start_day'] = d['発生時期']

    lane_order = ['末梢神経障害', '皮膚障害', '味覚異常', 'その他', '介入（休薬/減量）']
    lanes_present = [l for l in lane_order if l in d['lane'].unique().tolist()]
    if not lanes_present:
        lanes_present = sorted(d['lane'].unique().tolist())

    y_map = {lane: i for i, lane in enumerate(lanes_present)}
    y_ticks = [y_map[l] for l in lanes_present]
    y_labels = lanes_present

    fig_h = max(4, 0.8 * (len(lanes_present) + 1))
    fig, ax = plt.subplots(figsize=(12, fig_h))
    ax.set_facecolor(base_facecolor)

    # 背景帯
    for lane in lanes_present:
        y = y_map[lane]
        ax.barh(y, treatment_duration, color='#EAECEF', height=0.55, zorder=0)

    segments, seg_colors, seg_lws = [], [], []
    ae_points_by_marker = {}

    interventions = {
        '休薬': {'x': [], 'y': [], 'c': [], 's': [], 'marker': 'X', 'edge': []},
        '減量': {'x': [], 'y': [], 'c': [], 's': [], 'marker': 'v', 'edge': []},
    }
    intervention_texts = []
    ongoing_arrows = []

    # 同レーン同日の軽量重なり回避
    d['_lane_day_rank'] = d.groupby(['lane', '_start_day']).cumcount()
    d['_lane_day_count'] = d.groupby(['lane', '_start_day'])['_lane_day_rank'].transform('max') + 1

    for _, row in d.iterrows():
        start = row.get('発生時期', np.nan)
        if pd.isna(start):
            continue

        event_type = str(row.get('イベント種別', ''))
        lane = str(row.get('lane', 'その他'))
        y0 = float(y_map.get(lane, 0))

        rank = int(row.get('_lane_day_rank', 0))
        cnt = int(row.get('_lane_day_count', 1))

        y = y0
        if cnt > 1:
            k = rank + 1
            sign = 1 if (k % 2 == 1) else -1
            mag = 0.08 * ((k + 1) // 2)
            y = y0 + sign * min(0.18, mag)

        # 介入
        if event_type in interventions:
            if event_type == '休薬':
                c, edge, size = 'black', 'black', 160
            else:
                c, edge, size = 'blue', 'blue', 120

            interventions[event_type]['x'].append(start)
            interventions[event_type]['y'].append(y)
            interventions[event_type]['c'].append(c)
            interventions[event_type]['edge'].append(edge)
            interventions[event_type]['s'].append(size)

            if show_intervention_text:
                intervention_texts.append((start, y + 0.12, event_type, c))
            continue

        # AE期間
        grade = row.get('Grade', np.nan)
        end = row.get('消失時期', np.nan)
        ongoing = is_ongoing_value(row.get('継続中', '')) if '継続中' in d.columns else False
        if pd.isna(end) or ongoing:
            end = treatment_duration
            ongoing = True
        if pd.isna(end) or end < start:
            continue

        color = get_color_by_grade(grade)
        line_color = '#C0C0C0' if (pd.notna(grade) and int(grade) == 0) else color

        segments.append([(start, y), (end, y)])
        seg_colors.append(line_color)
        seg_lws.append(float(line_width))

        mk = marker_for_ae(event_type)
        if mk not in ae_points_by_marker:
            ae_points_by_marker[mk] = {'x': [], 'y': [], 'face': [], 'edge': [], 's': []}

        face = color
        edge = 'black'
        if pd.notna(grade) and int(grade) == 0:
            face = '#FFFFFF'
            edge = '#666666'

        base_size = 220 if mk == '*' else 140
        ae_points_by_marker[mk]['x'].append(start)
        ae_points_by_marker[mk]['y'].append(y)
        ae_points_by_marker[mk]['face'].append(face)
        ae_points_by_marker[mk]['edge'].append(edge)
        ae_points_by_marker[mk]['s'].append(base_size)

        if ongoing and show_ongoing_arrow:
            ongoing_arrows.append((end + 1, y, color))

    # Draw segments
    if segments:
        lc = LineCollection(
            segments,
            colors=seg_colors,
            linewidths=seg_lws,
            alpha=0.85,
            zorder=2
        )
        ax.add_collection(lc)

    # Draw AE points
    for mk, dd in ae_points_by_marker.items():
        ax.scatter(
            dd['x'], dd['y'],
            s=dd['s'],
            marker=mk,
            c=dd['face'],
            edgecolors=dd['edge'],
            linewidths=0.9,
            zorder=3
        )

    # Draw interventions
    for _, dd in interventions.items():
        if not dd['x']:
            continue
        ax.scatter(
            dd['x'], dd['y'],
            s=dd['s'],
            marker=dd['marker'],
            c=dd['c'],
            edgecolors=dd['edge'],
            linewidths=1.0,
            zorder=4
        )

    if show_intervention_text:
        for x, y, txt, c in intervention_texts:
            ax.text(x, y, txt, ha='center', va='bottom', fontsize=9, color=c, fontweight='bold', zorder=6)

    for x, y, c in ongoing_arrows:
        ax.text(x, y, '→', va='center', ha='left', fontsize=14, color=c, fontweight='bold', zorder=5)

    ax.set_title(f"単一患者スイムレーン（患者ID: {patient_id}）")
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel('投与開始からの期間（日）')
    ax.xaxis.grid(True, linestyle='--', alpha=0.5, zorder=0)
    ax.set_xlim(left=0, right=float(treatment_duration) + float(x_margin))

    # 上から末梢→皮膚→味覚→その他→介入の順に見せたい
    ax.invert_yaxis()

    # Legends
    if show_legends:
        legend_colors = [
            mlines.Line2D([], [], color='#FF4B4B', marker='o', linestyle='-', linewidth=3, markersize=8, label='G3以上'),
            mlines.Line2D([], [], color='#FFA07A', marker='o', linestyle='-', linewidth=3, markersize=8, label='G2'),
            mlines.Line2D([], [], color='#FFE4B5', marker='o', linestyle='-', linewidth=3, markersize=8, label='G1'),
            mlines.Line2D([], [], color='#C0C0C0', marker='o', markerfacecolor='#FFFFFF', markeredgecolor='#666666',
                          linestyle='-', linewidth=3, markersize=8, label='G0(回復)'),
        ]

        legend_shapes = [
            mlines.Line2D([], [], color='black', marker='*', linestyle='None', markersize=10, label='末梢神経障害'),
            mlines.Line2D([], [], color='black', marker='^', linestyle='None', markersize=10, label='皮膚障害'),
            mlines.Line2D([], [], color='black', marker='D', linestyle='None', markersize=10, label='味覚異常'),
            mlines.Line2D([], [], color='black', marker='o', linestyle='None', markersize=10, label='その他副作用'),
            mlines.Line2D([], [], color='black', marker='X', linestyle='None', markersize=10, label='休薬'),
            mlines.Line2D([], [], color='blue',  marker='v', linestyle='None', markersize=10, label='減量'),
        ]

        first_legend = ax.legend(
            handles=legend_colors,
            title='【重症度（色）】',
            loc='upper center',
            bbox_to_anchor=(0.5, -0.12),
            ncol=4,
            frameon=False,
            handlelength=2.5,
            columnspacing=1.2
        )
        ax.add_artist(first_legend)

        ax.legend(
            handles=legend_shapes,
            title='【イベント種別（形）】',
            loc='upper center',
            bbox_to_anchor=(0.5, -0.25),
            ncol=min(6, len(legend_shapes)),
            frameon=False,
            handlelength=1.5,
            columnspacing=1.2
        )

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.30)
    return fig


# =========================
# Main
# =========================
if uploaded_file is None:
    st.info('CSVファイルをアップロードすると、ここにグラフが表示されます。')
    st.stop()

df = pd.read_csv(uploaded_file)

# ---- preprocessing ----
numeric_cols = ['発生時期', '消失時期', '治療期間', 'Grade']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

if '継続中' in df.columns:
    df['継続中'] = df['継続中'].astype(str)

required_cols = {'患者ID', '治療期間', 'イベント種別', '発生時期'}
missing = required_cols - set(df.columns)
if missing:
    st.error(f"必要な列が不足しています: {', '.join(sorted(missing))}")
    st.stop()

treatment_map = df.groupby('患者ID')['治療期間'].max().to_dict()

st.success('データの読み込みに成功しました！')

with st.expander("【入力データプレビュー（クリックで展開）】"):
    st.dataframe(df)

# =========================
# Sidebar UI
# =========================
st.sidebar.header("表示設定（薬剤師向け）")

st.sidebar.subheader("表示モード")
view_mode = st.sidebar.radio(
    "mode",
    ["swimmer", "swimlane"],
    index=0,
    help="swimmer: 全患者 / swimlane: 単一患者（カテゴリ別）"
)

sort_mode = st.sidebar.selectbox(
    "患者の並び替え",
    ["患者ID（昇順）", "治療期間（長い順）", "治療期間（短い順）", "イベント数（多い順）", "最大Grade（高い順）"],
    index=0,
    help="学会図でよく使う並べ方を選べます。"
)

# Filter: patients
all_patients = sorted(df['患者ID'].dropna().unique().tolist())
selected_patients = st.sidebar.multiselect(
    "患者フィルタ（未選択なら全表示）",
    options=all_patients,
    default=[],
    help="特定の患者だけ確認したいときに使います。"
)

# Filter: events
all_event_types = sorted(df['イベント種別'].astype(str).dropna().unique().tolist())
selected_event_types = st.sidebar.multiselect(
    "イベント種別フィルタ（未選択なら全表示）",
    options=all_event_types,
    default=[],
    help="特定の副作用や介入（休薬/減量）だけに絞り込めます。"
)

# Grade filter
grade_min, grade_max = st.sidebar.slider(
    "Gradeフィルタ（副作用のみ）",
    min_value=0,
    max_value=4,
    value=(0, 4),
    help="副作用（AE）のGradeだけを絞り込みます。休薬/減量など介入は対象外です。"
)

# Visual toggles
show_interventions = st.sidebar.checkbox("介入イベント（休薬/減量）を表示", value=True)
show_ongoing_arrow = st.sidebar.checkbox("継続中（→）を表示", value=True, help="継続中の副作用を矢印で示します。")
show_intervention_text = st.sidebar.checkbox("介入テキスト（休薬/減量）を表示", value=False, help="文字表示は密集時に読みにくくなることがあります。")
show_legends = st.sidebar.checkbox("凡例（色/形の説明）を表示", value=True)

# ---- Simple clinician-friendly layout controls ----
st.sidebar.subheader("見え方の調整（かんたん）")

clarity = st.sidebar.slider(
    "表示の見やすさ",
    0, 100, 40,
    help="左：全体を俯瞰（コンパクト）／右：1人の患者を読みやすく表示（詳細）"
)

overlap_strength = st.sidebar.slider(
    "重なり回避の強さ",
    0, 100, 60,
    help="同じ日に複数イベントがある場合、マーカーを上下に分散して見やすくします。"
)

EXPAND_THRESHOLD, MAX_LANES, gap, LANE_INNER_JITTER_STEP, LANE_INNER_JITTER_MAX = preset_to_layout(clarity)
LANE_INNER_JITTER_STEP *= (0.6 + 0.8 * (overlap_strength / 100))
LANE_INNER_JITTER_MAX *= (0.6 + 0.8 * (overlap_strength / 100))

with st.sidebar.expander("詳細設定（必要な場合のみ）"):
    st.caption("※通常は触らなくてOKです。図が読みにくいときだけ調整してください。")
    EXPAND_THRESHOLD = st.slider(
        "行を広げる条件（同じ日にイベントが何個以上なら広げるか）",
        2, 8, EXPAND_THRESHOLD,
        help="例：3なら『同日3イベント以上』の患者だけ、行が段数化されて見やすくなります。"
    )
    MAX_LANES = st.slider(
        "1人の患者で使える最大の段数",
        1, 10, MAX_LANES,
        help="段数を増やすと重なりに強くなりますが、全体が縦に長くなります。"
    )
    gap = st.slider(
        "患者と患者の間隔",
        0.1, 2.0, gap,
        help="詰めると俯瞰しやすく、広げると読みやすくなります。"
    )
    LANE_INNER_JITTER_STEP = st.slider(
        "段数を超えたときの追加のずらし幅",
        0.0, 0.12, float(LANE_INNER_JITTER_STEP),
        help="極端に同日イベントが多い場合の保険です。通常はそのままでOK。"
    )
    LANE_INNER_JITTER_MAX = st.slider(
        "追加のずらしの上限",
        0.0, 0.20, float(LANE_INNER_JITTER_MAX),
        help="ずらしすぎて他の段と混ざらないよう、上限を決めています。"
    )

# Figure look
st.sidebar.subheader("図の見た目")
line_width = st.sidebar.slider("期間線の太さ", 1, 6, 3, help="副作用の期間を示す線の太さです。")
base_facecolor = st.sidebar.selectbox("背景色", ["#F8F9FA", "white"], index=0)
x_margin = st.sidebar.slider("右余白（日）", 0, 30, 10, help="継続中（→）が切れないように右側へ余白を足します。")

# =========================
# Apply filters
# =========================
df_plot = df.copy()

if selected_patients:
    df_plot = df_plot[df_plot['患者ID'].isin(selected_patients)]

if selected_event_types:
    df_plot = df_plot[df_plot['イベント種別'].astype(str).isin(selected_event_types)]

is_intervention = df_plot['イベント種別'].astype(str).isin(['休薬', '減量'])

if show_interventions:
    ae_mask = ~is_intervention
    within = df_plot['Grade'].between(grade_min, grade_max, inclusive='both')
    df_plot = df_plot[(is_intervention) | (ae_mask & within)]
else:
    df_plot = df_plot[~is_intervention]
    df_plot = df_plot[df_plot['Grade'].between(grade_min, grade_max, inclusive='both')]

if df_plot.empty:
    st.warning("フィルタ条件に一致するデータがありません。")
    st.stop()

treatment_map_plot = df_plot.groupby('患者ID')['治療期間'].max().to_dict()

# =========================
# Build plotting layout (swimmer)
# =========================
patients = stable_patients_order(df_plot, sort_mode, treatment_map_plot)

event_priority_map = {'休薬': 0, '減量': 1}
df_plot['_event_priority'] = df_plot['イベント種別'].astype(str).map(event_priority_map).fillna(2)
df_plot['_start_day'] = df_plot['発生時期']

df_plot = df_plot.sort_values(
    by=['患者ID', '_start_day', '_event_priority', 'イベント種別', 'Grade'],
    kind='mergesort'
)

df_plot['_same_day_rank'] = df_plot.groupby(['患者ID', '_start_day']).cumcount()
df_plot['_same_day_count'] = df_plot.groupby(['患者ID', '_start_day'])['_same_day_rank'].transform('max') + 1

pt_max_same_day = (
    df_plot.groupby('患者ID')['_same_day_count']
      .max()
      .reindex(patients)
      .fillna(1)
      .astype(int)
      .to_dict()
)

pt_lanes = {}
for pt in patients:
    m = pt_max_same_day.get(pt, 1)
    if m >= EXPAND_THRESHOLD:
        pt_lanes[pt] = int(min(MAX_LANES, max(1, m)))
    else:
        pt_lanes[pt] = 1

y_center_map = {}
pt_lane_y_map = {}
cursor = 0.0

for pt in patients:
    lanes = pt_lanes[pt]
    lane_ys = cursor + np.arange(lanes, dtype=float)
    center_y = float(lane_ys.mean())
    y_center_map[pt] = center_y
    for li, yy in enumerate(lane_ys):
        pt_lane_y_map[(pt, li)] = float(yy)
    cursor = float(lane_ys.max() + 1.0 + gap)

yticks = [y_center_map[pt] for pt in patients]
yticklabels = patients

# =========================
# Figure (swimmer)
# =========================
fig_h = max(4, 0.45 * (cursor + 1))
fig, ax = plt.subplots(figsize=(12, fig_h))
ax.set_facecolor(base_facecolor)

# baseline bars
for pt in patients:
    duration = treatment_map_plot.get(pt, 0)
    lanes = pt_lanes[pt]
    y = y_center_map[pt]
    height = min(0.90, 0.15 + 0.18 * (lanes - 1))
    ax.barh(y, duration, color='#E0E0E0', height=height, zorder=1)

segments = []
seg_colors = []
seg_lws = []

ae_points_by_marker = {}

interventions = {
    '休薬': {'x': [], 'y': [], 'c': [], 's': [], 'marker': 'X', 'edge': []},
    '減量': {'x': [], 'y': [], 'c': [], 's': [], 'marker': 'v', 'edge': []},
}
intervention_texts = []
ongoing_arrows = []

for _, row in df_plot.iterrows():
    pt = row['患者ID']
    if pt not in y_center_map:
        continue

    start = row['発生時期']
    if pd.isna(start):
        continue

    event_type = str(row['イベント種別'])
    grade = row.get('Grade', np.nan)

    same_day_rank = int(row.get('_same_day_rank', 0)) if pd.notna(row.get('_same_day_rank', 0)) else 0
    same_day_count = int(row.get('_same_day_count', 1)) if pd.notna(row.get('_same_day_count', 1)) else 1

    lanes = pt_lanes[pt]
    lane_index = same_day_rank % lanes
    y = pt_lane_y_map[(pt, lane_index)]

    overflow_group = same_day_rank // lanes
    if overflow_group > 0 and LANE_INNER_JITTER_STEP > 0:
        sign = 1 if (overflow_group % 2 == 1) else -1
        mag = min(LANE_INNER_JITTER_MAX, (1 + (overflow_group // 2)) * LANE_INNER_JITTER_STEP)
        y = y + sign * mag

    scale = marker_scale_for_count(same_day_count)
    ongoing = is_ongoing_value(row.get('継続中', '')) if '継続中' in df_plot.columns else False

    # interventions
    if event_type in interventions:
        if not show_interventions:
            continue
        if event_type == '休薬':
            c, edge, size = 'black', 'black', 160 * scale
        else:
            c, edge, size = 'blue', 'blue', 120 * scale

        interventions[event_type]['x'].append(start)
        interventions[event_type]['y'].append(y)
        interventions[event_type]['c'].append(c)
        interventions[event_type]['edge'].append(edge)
        interventions[event_type]['s'].append(size)

        if show_intervention_text:
            intervention_texts.append((start, y + 0.18, event_type, c))

        continue

    # AE
    end = row.get('消失時期', np.nan)
    if pd.isna(end) or ongoing:
        end = treatment_map_plot.get(pt, start)
        ongoing = True

    if pd.isna(end) or end < start:
        continue

    color = get_color_by_grade(grade)
    line_color = '#C0C0C0' if (pd.notna(grade) and int(grade) == 0) else color

    segments.append([(start, y), (end, y)])
    seg_colors.append(line_color)
    seg_lws.append(float(line_width))

    mk = marker_for_ae(event_type)
    if mk not in ae_points_by_marker:
        ae_points_by_marker[mk] = {'x': [], 'y': [], 'face': [], 'edge': [], 's': []}

    base_size = 220 if mk == '*' else 140
    size = base_size * scale

    face = color
    edge = 'black'
    if pd.notna(grade) and int(grade) == 0:
        face = '#FFFFFF'
        edge = '#666666'

    ae_points_by_marker[mk]['x'].append(start)
    ae_points_by_marker[mk]['y'].append(y)
    ae_points_by_marker[mk]['face'].append(face)
    ae_points_by_marker[mk]['edge'].append(edge)
    ae_points_by_marker[mk]['s'].append(size)

    if ongoing and show_ongoing_arrow:
        ongoing_arrows.append((end + 1, y, color))

# Draw
if segments:
    lc = LineCollection(
        segments,
        colors=seg_colors,
        linewidths=seg_lws,
        alpha=0.85,
        zorder=2
    )
    ax.add_collection(lc)

for mk, ddd in ae_points_by_marker.items():
    ax.scatter(
        ddd['x'], ddd['y'],
        s=ddd['s'],
        marker=mk,
        c=ddd['face'],
        edgecolors=ddd['edge'],
        linewidths=0.9,
        zorder=3
    )

if show_interventions:
    for _, ddd in interventions.items():
        if not ddd['x']:
            continue
        ax.scatter(
            ddd['x'], ddd['y'],
            s=ddd['s'],
            marker=ddd['marker'],
            c=ddd['c'],
            edgecolors=ddd['edge'],
            linewidths=1.0,
            zorder=4
        )

if show_intervention_text:
    for x, y, txt, c in intervention_texts:
        ax.text(x, y, txt, ha='center', va='bottom', fontsize=9, color=c, fontweight='bold', zorder=6)

for x, y, c in ongoing_arrows:
    ax.text(x, y, '→', va='center', ha='left', fontsize=14, color=c, fontweight='bold', zorder=5)

# Axes
ax.set_yticks(yticks)
ax.set_yticklabels(yticklabels)
ax.set_xlabel('投与開始からの期間（日）')
ax.xaxis.grid(True, linestyle='--', alpha=0.5, zorder=0)

max_duration = max([treatment_map_plot.get(pt, 0) for pt in patients] + [0])
ax.set_xlim(left=0, right=max_duration + x_margin)

# Legends
if show_legends:
    legend_colors = [
        mlines.Line2D([], [], color='#FF4B4B', marker='o', linestyle='-', linewidth=3, markersize=8, label='G3以上'),
        mlines.Line2D([], [], color='#FFA07A', marker='o', linestyle='-', linewidth=3, markersize=8, label='G2'),
        mlines.Line2D([], [], color='#FFE4B5', marker='o', linestyle='-', linewidth=3, markersize=8, label='G1'),
        mlines.Line2D([], [], color='#C0C0C0', marker='o', markerfacecolor='#FFFFFF', markeredgecolor='#666666',
                      linestyle='-', linewidth=3, markersize=8, label='G0(回復)'),
    ]

    legend_shapes = [
        mlines.Line2D([], [], color='black', marker='*', linestyle='None', markersize=10, label='末梢神経障害'),
        mlines.Line2D([], [], color='black', marker='^', linestyle='None', markersize=10, label='皮膚障害'),
        mlines.Line2D([], [], color='black', marker='D', linestyle='None', markersize=10, label='味覚異常'),
        mlines.Line2D([], [], color='black', marker='o', linestyle='None', markersize=10, label='その他副作用'),
    ]
    if show_interventions:
        legend_shapes += [
            mlines.Line2D([], [], color='black', marker='X', linestyle='None', markersize=10, label='休薬'),
            mlines.Line2D([], [], color='blue',  marker='v', linestyle='None', markersize=10, label='減量'),
        ]

    first_legend = ax.legend(
        handles=legend_colors,
        title='【重症度（色）】',
        loc='upper center',
        bbox_to_anchor=(0.5, -0.12),
        ncol=4,
        frameon=False,
        handlelength=2.5,
        columnspacing=1.2
    )
    ax.add_artist(first_legend)

    ax.legend(
        handles=legend_shapes,
        title='【イベント種別（形）】',
        loc='upper center',
        bbox_to_anchor=(0.5, -0.25),
        ncol=min(6, len(legend_shapes)),
        frameon=False,
        handlelength=1.5,
        columnspacing=1.2
    )

plt.tight_layout()
fig.subplots_adjust(bottom=0.30)

# =========================
# Render (mode switch)
# =========================
if view_mode == "swimmer":
    st.pyplot(fig)
else:
    st.sidebar.subheader("患者選択（swimlane）")
    if not patients:
        st.warning("表示可能な患者がいません。")
        st.stop()

    selected_pt = st.sidebar.selectbox("患者ID", options=patients, index=0)
    df_pt = df_plot[df_plot['患者ID'] == selected_pt].copy()
    tdur = float(treatment_map_plot.get(selected_pt, df_pt['治療期間'].max() if '治療期間' in df_pt.columns else 0))

    fig2 = plot_single_patient_swimlanes(
        df_pt=df_pt,
        patient_id=selected_pt,
        treatment_duration=tdur,
        line_width=line_width,
        base_facecolor=base_facecolor,
        x_margin=x_margin,
        show_legends=show_legends,
        show_ongoing_arrow=show_ongoing_arrow,
        show_intervention_text=show_intervention_text,
    )
    st.pyplot(fig2)

# =========================
# Optional: diagnostic panel
# =========================
with st.expander("【診断情報（開発者向け）】"):
    st.write("患者ごとの最大同日イベント数 / 割当段数（図が縦に伸びる理由の確認用）")
    diag = pd.DataFrame({
        '患者ID': patients,
        '最大同日イベント数': [pt_max_same_day.get(p, 1) for p in patients],
        '割当段数': [pt_lanes.get(p, 1) for p in patients],
        '治療期間': [treatment_map_plot.get(p, np.nan) for p in patients],
        'イベント数': [int(df_plot[df_plot['患者ID'] == p].shape[0]) for p in patients],
    })
    st.dataframe(diag)