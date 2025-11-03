"""
频谱分析模块 - 基于consistency_app.py
实现新数据与参考集的频谱对比分析，包含完整的测点切换功能
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io
import re
import zipfile

# 页面配置
st.set_page_config(
    page_title="频谱分析 - 力学振动数据一致性分析系统",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 应用标题
st.title("📊 频谱分析模块")
st.markdown("新数据 vs 参考集频谱对比分析")
st.markdown("---")

# 转换函数：与 yizhixing/lms_converter.py 逻辑一致
@st.cache_data
def convert_lms_excel(file):
    """
    - 读取第一个工作表（不设header）
    - 第12行（索引11）获取测点名称，取偶数列（索引1,3,5...）
    - 第一列为频率（索引0），数据从第13行（索引12）开始
    - 测点名保留 XM/YM/ZM + 数字（如 XM1、YM2、ZM3）
    - 输出 DataFrame: 第一列 'HZ' + 测点列
    """
    try:
        excel_file = pd.ExcelFile(file)
        sheet_names = excel_file.sheet_names
        df = pd.read_excel(file, sheet_name=sheet_names[0], header=None)

        row12 = df.iloc[11]
        measurement_indices = [i for i in range(1, len(row12), 2) if pd.notna(row12[i])]
        measurement_names = [str(row12[i]).strip() for i in measurement_indices]

        frequency_col = 0
        data_start_row = 12
        df_data = df.iloc[data_start_row:]

        processed_df = pd.DataFrame()
        processed_df['HZ'] = df_data[frequency_col].reset_index(drop=True)

        for idx, name in zip(measurement_indices, measurement_names):
            match = re.search(r'(XM\d+|YM\d+|ZM\d+)', name)
            point_name = match.group(1) if match else name
            processed_df[point_name] = df_data[idx].reset_index(drop=True)

        processed_df = processed_df.dropna(how='all')
        processed_df['HZ'] = pd.to_numeric(processed_df['HZ'], errors='coerce')
        for col in processed_df.columns[1:]:
            processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
        processed_df = processed_df.dropna()
        return processed_df
    except Exception:
        return None

@st.cache_data
def df_to_excel_bytes(df, sheet_name='转换数据'):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
    buf.seek(0)
    return buf

@st.cache_data
def read_processed_excel(file):
    """
    读取已处理参考数据（第一列为频率，其他列为测点）。
    """
    try:
        excel_file = pd.ExcelFile(file)
        sheet_name = excel_file.sheet_names[0]
        df = pd.read_excel(file, sheet_name=sheet_name)
        df.iloc[:, 0] = pd.to_numeric(df.iloc[:, 0], errors='coerce')
        for c in df.columns[1:]:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df = df.dropna()
        return df
    except Exception:
        return None

@st.cache_data
def load_location_data(_location_file):
    """加载测点位置表数据"""
    try:
        # 读取位置表
        location_df = pd.read_excel(_location_file)
        
        # 检查必要的列是否存在
        required_columns = ['测点名称', '舱板', '单机区域']
        missing_columns = [col for col in required_columns if col not in location_df.columns]
        
        if missing_columns:
            st.error(f"测点位置表缺少必要的列: {missing_columns}")
            return {}
        
        # 创建测点位置映射字典
        location_data = {}
        for _, row in location_df.iterrows():
            point_name = str(row['测点名称']).strip()
            cabin = str(row['舱板']).strip()
            area = str(row['单机区域']).strip()
            location_data[point_name] = f"{cabin}-{area}"
        
        return location_data
        
    except Exception as e:
        st.error(f"测点位置表读取错误: {str(e)}")
        return {}

def get_point_location(point_name):
    """获取测点的位置信息"""
    # 从测点名称中提取数字部分（例如：从"XM1"中提取"1"）
    match = re.search(r'\d+', point_name)
    if match:
        point_number = match.group()  # 提取到的数字（字符串格式）
        
        # 用数字去位置表中查找
        if point_number in st.session_state.location_data:
            return st.session_state.location_data[point_number]
    
    return "未知位置"

def format_point_with_location(point_name):
    """格式化测点信息，添加位置信息"""
    location = get_point_location(point_name)
    return f"{point_name} ({location})"

def extract_measurement_points(dfs):
    points = set()
    for df in dfs.values():
        if df is not None and df.shape[1] >= 2:
            pts = [c for c in df.columns[1:]]
            points.update(pts)
    return points

def create_spectrum_plot_emphasis(new_dict, ref_dict, selected_point,
                                  ref_opacity=0.75, ref_line_width=1.6, ref_dash='dash',
                                  new_line_width=2.5):
    fig = go.Figure()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#bcbd22', '#17becf']

    # 参考数据淡化（但保持清晰）
    for label, df in ref_dict.items():
        if df is None or selected_point not in df.columns:
            continue
        fig.add_trace(go.Scatter(
            x=df.iloc[:, 0], y=df[selected_point], mode='lines',
            name=f"参考-{label}", line=dict(color='#7f7f7f', width=ref_line_width, dash=ref_dash),
            opacity=ref_opacity,
            hovertemplate=(
                "<b>频率</b>: %{x:.2f} Hz<br>"
                "<b>幅值</b>: %{y:.4f}<br>"
                f"<b>数据集</b>: 参考-{label}<br>"
                f"<b>测点</b>: {selected_point}<extra></extra>"
            )
        ))

    # 新数据高亮
    for i, (label, df) in enumerate(new_dict.items()):
        if df is None or selected_point not in df.columns:
            continue
        color = colors[i % len(colors)]
        fig.add_trace(go.Scatter(
            x=df.iloc[:, 0], y=df[selected_point], mode='lines',
            name=f"新-{label}", line=dict(color=color, width=new_line_width),
            hovertemplate=(
                "<b>频率</b>: %{x:.2f} Hz<br>"
                "<b>幅值</b>: %{y:.4f}<br>"
                f"<b>数据集</b>: 新-{label}<br>"
                f"<b>测点</b>: {selected_point}<extra></extra>"
            )
        ))

    fig.update_layout(
        title=f"频谱分析（测点：{selected_point}）",
        xaxis_title="频率 (Hz)", yaxis_title="响应幅值",
        height=600,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode='closest'
    )
    fig.update_xaxes(type="log", gridcolor='lightgray', gridwidth=1, showgrid=True)
    fig.update_yaxes(type="log", gridcolor='lightgray', gridwidth=1, showgrid=True)
    return fig

def compute_resonance_in_band(df, point, fmin, fmax):
    if df is None or point not in df.columns:
        return None, None
    freqs = df.iloc[:, 0].values
    amps = df[point].values
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        return None, None
    sub_freqs = freqs[mask]
    sub_amps = amps[mask]
    if len(sub_amps) == 0:
        return None, None
    idx = np.argmax(sub_amps)
    return float(sub_freqs[idx]), float(sub_amps[idx])

def plot_band_overlay(new_dict, ref_dict, selected_point, fmin, fmax,
                      ref_opacity=0.75, ref_line_width=1.6, ref_dash='solid',
                      new_line_width=2.5, axis_lower=0.0, axis_upper=150.0):
    fig = go.Figure()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#bcbd22', '#17becf']

    # 参考数据：分段绘制（频段外更淡，频段内更清晰）
    for label, df in ref_dict.items():
        if df is None or selected_point not in df.columns:
            continue
        F = df.iloc[:, 0].values
        A = df[selected_point].values
        mask = (F >= fmin) & (F <= fmax)
        fig.add_trace(go.Scatter(
            x=F[~mask], y=A[~mask], mode='lines', name=f"参考-{label}(频段外)",
            line=dict(color='#b0b0b0', width=max(0.8, ref_line_width*0.7), dash='dot'),
            opacity=max(0.15, ref_opacity*0.4), showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=F[mask], y=A[mask], mode='lines', name=f"参考-{label}",
            line=dict(color='#7f7f7f', width=ref_line_width, dash=ref_dash), opacity=ref_opacity
        ))

    # 新数据：分段绘制（频段内高亮）
    for i, (label, df) in enumerate(new_dict.items()):
        if df is None or selected_point not in df.columns:
            continue
        color = colors[i % len(colors)]
        F = df.iloc[:, 0].values
        A = df[selected_point].values
        mask = (F >= fmin) & (F <= fmax)
        fig.add_trace(go.Scatter(
            x=F[~mask], y=A[~mask], mode='lines', name=f"新-{label}(频段外)",
            line=dict(color=color, width=max(1.0, new_line_width*0.6), dash='dot'),
            opacity=0.4, showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=F[mask], y=A[mask], mode='lines', name=f"新-{label}",
            line=dict(color=color, width=new_line_width)
        ))

    # 参考平均频率（选定频段共振频率的均值）
    ref_res_freqs = []
    for label, df in ref_dict.items():
        rf, ra = compute_resonance_in_band(df, selected_point, fmin, fmax)
        if rf is not None:
            ref_res_freqs.append(rf)
    ref_avg_freq = float(np.mean(ref_res_freqs)) if ref_res_freqs else None
    if ref_avg_freq is not None:
            fig.add_vline(x=ref_avg_freq, line=dict(color='black', width=2, dash='dash'),
                      annotation_text=f"参考均值 {ref_avg_freq:.2f}Hz", annotation_position="top")

    # 新数据各文件共振频率
    for i, (label, df) in enumerate(new_dict.items()):
        rf, ra = compute_resonance_in_band(df, selected_point, fmin, fmax)
        if rf is not None:
            color = colors[i % len(colors)]
            fig.add_vline(x=rf, line=dict(color=color, width=1, dash='dot'),
                          annotation_text=f"新-{label} {rf:.2f}Hz", annotation_position="top")

    fig.update_layout(
        title=f"选定频段频谱叠加（测点：{selected_point}，{fmin:.2f}-{fmax:.2f}Hz）",
        xaxis_title="频率 (Hz)", yaxis_title="响应幅值",
        height=600,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode='closest'
    )
    fig.update_xaxes(type="linear", range=[axis_lower, axis_upper], gridcolor='lightgray', gridwidth=1, showgrid=True)
    fig.update_yaxes(type="log", gridcolor='lightgray', gridwidth=1, showgrid=True)
    return fig

def compare_new_vs_ref_in_band(new_dict, ref_dict, selected_point, fmin, fmax):
    # 参考均值（频率与幅值）
    ref_freqs, ref_amps = [], []
    for label, df in ref_dict.items():
        rf, ra = compute_resonance_in_band(df, selected_point, fmin, fmax)
        if rf is not None and ra is not None:
            ref_freqs.append(rf)
            ref_amps.append(ra)
    ref_avg_freq = float(np.mean(ref_freqs)) if ref_freqs else None
    ref_avg_amp  = float(np.mean(ref_amps)) if ref_amps else None

    rows = []
    for label, df in new_dict.items():
        new_rf, new_ra = compute_resonance_in_band(df, selected_point, fmin, fmax)
        if new_rf is None or new_ra is None or ref_avg_freq is None or ref_avg_amp is None:
            rows.append({
                '新数据文件': label,
                '测点': selected_point,
                '新共振频率(Hz)': new_rf if new_rf is not None else 'N/A',
                '新幅值': new_ra if new_ra is not None else 'N/A',
                '参考平均频率(Hz)': ref_avg_freq if ref_avg_freq is not None else 'N/A',
                '参考平均幅值': ref_avg_amp if ref_avg_amp is not None else 'N/A',
                '频率差(新-参考均值)Hz': 'N/A',
                '幅值差(新-参考均值)': 'N/A'
            })
            continue
        rows.append({
            '新数据文件': label,
            '测点': selected_point,
            '新共振频率(Hz)': new_rf,
            '新幅值': new_ra,
            '参考平均频率(Hz)': ref_avg_freq,
            '参考平均幅值': ref_avg_amp,
            '频率差(新-参考均值)Hz': float(new_rf - ref_avg_freq),
            '幅值差(新-参考均值)': float(new_ra - ref_avg_amp),
        })
    result_df = pd.DataFrame(rows)
    return result_df, ref_avg_freq, ref_avg_amp

def compare_all_points_in_band(new_dict, ref_dict, all_points, fmin, fmax):
    """
    计算所有测点的差值比较结果
    """
    all_rows = []
    
    for selected_point in all_points:
        # 参考均值（频率与幅值）- 每个测点单独计算
        ref_freqs, ref_amps = [], []
        for label, df in ref_dict.items():
            rf, ra = compute_resonance_in_band(df, selected_point, fmin, fmax)
            if rf is not None and ra is not None:
                ref_freqs.append(rf)
                ref_amps.append(ra)
        ref_avg_freq = float(np.mean(ref_freqs)) if ref_freqs else None
        ref_avg_amp  = float(np.mean(ref_amps)) if ref_amps else None

        # 新数据各文件
        for label, df in new_dict.items():
            new_rf, new_ra = compute_resonance_in_band(df, selected_point, fmin, fmax)
            if new_rf is None or new_ra is None or ref_avg_freq is None or ref_avg_amp is None:
                all_rows.append({
                    '新数据文件': label,
                    '测点': selected_point,
                    '新共振频率(Hz)': new_rf if new_rf is not None else 'N/A',
                    '新幅值': new_ra if new_ra is not None else 'N/A',
                    '参考平均频率(Hz)': ref_avg_freq if ref_avg_freq is not None else 'N/A',
                    '参考平均幅值': ref_avg_amp if ref_avg_amp is not None else 'N/A',
                    '频率差(新-参考均值)Hz': 'N/A',
                    '幅值差(新-参考均值)': 'N/A'
                })
                continue
            all_rows.append({
                '新数据文件': label,
                '测点': selected_point,
                '新共振频率(Hz)': new_rf,
                '新幅值': new_ra,
                '参考平均频率(Hz)': ref_avg_freq,
                '参考平均幅值': ref_avg_amp,
                '频率差(新-参考均值)Hz': float(new_rf - ref_avg_freq),
                '幅值差(新-参考均值)': float(new_ra - ref_avg_amp),
            })
    
    result_df = pd.DataFrame(all_rows)
    return result_df

# 侧边栏：数据上传与转换
st.sidebar.header("📁 上传数据")
new_raw_files = st.sidebar.file_uploader(
    "上传新的未经处理的 LMS 数据（.xlsx/.xls，可多选）",
    type=['xlsx', 'xls'], accept_multiple_files=True,
    help="原始 LMS 导出（含第12行测点名），将自动转换为特征级数据。"
)
ref_files = st.sidebar.file_uploader(
    "上传参考数据（已处理特征级 Excel，第一列为频率，可多选）",
    type=['xlsx'], accept_multiple_files=True
)

# 侧边栏 - 测点位置表上传
st.sidebar.header("📍 测点位置信息")
location_file = st.sidebar.file_uploader(
    "选择测点位置表 (.xlsx)",
    type=["xlsx"],
    help="包含测点舱板和单机区域分布信息的Excel文件"
)

# 会话状态初始化
if 'new_processed_dict' not in st.session_state:
    st.session_state.new_processed_dict = {}
if 'ref_processed_dict' not in st.session_state:
    st.session_state.ref_processed_dict = {}
if 'conversion_buffers' not in st.session_state:
    st.session_state.conversion_buffers = {}
if 'current_point_index' not in st.session_state:
    st.session_state.current_point_index = 0
if 'current_point_index_band' not in st.session_state:
    st.session_state.current_point_index_band = 0
if 'location_data' not in st.session_state:
    st.session_state.location_data = {}
if 'location_file_loaded' not in st.session_state:
    st.session_state.location_file_loaded = False

# 转换新数据
st.sidebar.header("🔄 新数据转换")
if new_raw_files:
    if st.sidebar.button("开始转换并缓存", use_container_width=True):
        with st.spinner("正在转换新数据..."):
            st.session_state.new_processed_dict.clear()
            st.session_state.conversion_buffers.clear()
            for f in new_raw_files:
                df = convert_lms_excel(f)
                if df is not None and df.shape[1] >= 2:
                    st.session_state.new_processed_dict[f.name] = df
                    buf = df_to_excel_bytes(df, sheet_name='转换数据')
                    st.session_state.conversion_buffers[f.name] = buf.getvalue()
        st.sidebar.success(f"✅ 已转换 {len(st.session_state.new_processed_dict)} 个文件")

# 参考数据读取
st.sidebar.header("📘 参考数据读取")
if ref_files:
    if st.sidebar.button("读取参考数据", use_container_width=True):
        with st.spinner("正在读取参考数据..."):
            st.session_state.ref_processed_dict.clear()
            for f in ref_files:
                df = read_processed_excel(f)
                if df is not None and df.shape[1] >= 2:
                    st.session_state.ref_processed_dict[f.name] = df
        st.sidebar.success(f"✅ 已读取 {len(st.session_state.ref_processed_dict)} 个参考文件")

# 测点位置表读取
if location_file is not None:
    if not st.session_state.location_file_loaded or st.session_state.get('current_location_file') != location_file.name:
        with st.spinner("正在加载测点位置表..."):
            st.session_state.location_data = load_location_data(location_file)
            st.session_state.location_file_loaded = True
            st.session_state.current_location_file = location_file.name
            if st.session_state.location_data:
                st.sidebar.success(f"✅ 成功加载测点位置表: {location_file.name}")
                st.sidebar.info(f"📍 已加载 {len(st.session_state.location_data)} 个测点的位置信息")
            else:
                st.sidebar.warning("⚠️ 测点位置表加载失败或格式不正确")

# 显示风格控制
st.sidebar.header("🎨 显示风格")
ref_opacity = st.sidebar.slider("参考曲线不透明度", 0.2, 1.0, 0.75, 0.05)
ref_line_width = st.sidebar.slider("参考曲线线宽", 0.5, 3.0, 1.6, 0.1)
ref_line_style = st.sidebar.selectbox("参考曲线线型", ["solid", "dash", "dot"], index=1)
new_line_width = st.sidebar.slider("新数据线宽", 1.0, 4.0, 2.5, 0.1)

# 转换结果下载（单个）
if st.session_state.conversion_buffers:
    st.sidebar.header("📥 下载转换结果")
    for fname, b in st.session_state.conversion_buffers.items():
        st.sidebar.download_button(
            label=f"下载转换后：{fname}",
            data=b,
            file_name=f"converted_{fname}",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

# 频谱分析
st.subheader("① 频谱分析（新数据高亮，参考淡化）")
new_dict = st.session_state.new_processed_dict
ref_dict = st.session_state.ref_processed_dict

if not new_dict:
    st.info("请先上传并转换新的未经处理数据。")
else:
    new_points = extract_measurement_points(new_dict)
    ref_points = extract_measurement_points(ref_dict) if ref_dict else set()
    common_points = new_points & ref_points if ref_points else new_points

    if not common_points:
        st.warning("未找到可在新与参考之间共同叠加的测点。将仅显示新数据。")
        common_points = new_points

    sorted_points = sorted(
        list(common_points),
        key=lambda p: (p[:2], int(re.sub(r'\D', '', p) or 0))
    )
    if not sorted_points:
        st.error("数据中未检测到有效测点。")
    else:
        # 创建带位置信息的测点选项
        if st.session_state.location_data:
            point_options = [format_point_with_location(point) for point in sorted_points]
        else:
            point_options = sorted_points
        
        cols = st.columns([3, 1, 1])
        with cols[0]:
            selected_point_with_location = st.selectbox(
                "选择测点",
                point_options,
                index=min(st.session_state.current_point_index, len(point_options)-1)
            )
        
        # 提取原始测点名称
        if st.session_state.location_data:
            # 从带位置信息的选项中提取原始测点名称
            selected_point = selected_point_with_location.split(' (')[0]
        else:
            selected_point = selected_point_with_location
        with cols[1]:
            if st.button("⬅️ 上一个", use_container_width=True):
                st.session_state.current_point_index = (st.session_state.current_point_index - 1) % len(sorted_points)
                st.rerun()
        with cols[2]:
            if st.button("➡️ 下一个", use_container_width=True):
                st.session_state.current_point_index = (st.session_state.current_point_index + 1) % len(sorted_points)
                st.rerun()

        fig = create_spectrum_plot_emphasis(
            new_dict, ref_dict, selected_point,
            ref_opacity=ref_opacity, ref_line_width=ref_line_width, ref_dash=ref_line_style,
            new_line_width=new_line_width
        )
        st.plotly_chart(fig, use_container_width=True)
        st.download_button(
            label="📥 下载当前频谱图 (HTML)",
            data=fig.to_html(),
            file_name=f"spectrum_{selected_point}.html",
            mime="text/html"
        )

# 选定频段的频谱分析与差值比较
st.markdown("---")
st.subheader("② 选定频段频谱分析与差值比较（参考集只计算平均，新数据与其比较）")

if not new_dict:
    st.info("请先上传并转换新的未经处理数据。")
else:
    axis_lower, axis_upper = 0.0, 150.0
    new_points = extract_measurement_points(new_dict)
    ref_points = extract_measurement_points(ref_dict) if ref_dict else set()
    common_points = new_points & ref_points if ref_points else new_points

    if not common_points:
        st.warning("未找到可用于频段分析的测点。")
    else:
        sorted_points = sorted(
            list(common_points),
            key=lambda p: (p[:2], int(re.sub(r'\D', '', p) or 0))
        )
        st.session_state.current_point_index_band = min(st.session_state.current_point_index_band, len(sorted_points)-1)
        selected_point_band = sorted_points[st.session_state.current_point_index_band]

        cols = st.columns([2, 2, 2, 1, 1])
        with cols[0]:
            fmin = st.number_input("起始频率 (Hz)", min_value=axis_lower, max_value=axis_upper, value=axis_lower, step=1.0, format="%.3f")
        with cols[1]:
            fmax = st.number_input("终止频率 (Hz)", min_value=axis_lower, max_value=axis_upper, value=axis_upper, step=1.0, format="%.3f")
        with cols[2]:
            st.markdown(f"当前测点：**{selected_point_band}**")
        with cols[3]:
            if st.button("⬅️ 上一个测点", use_container_width=True):
                st.session_state.current_point_index_band = (st.session_state.current_point_index_band - 1) % len(sorted_points)
                st.rerun()
        with cols[4]:
            if st.button("➡️ 下一个测点", use_container_width=True):
                st.session_state.current_point_index_band = (st.session_state.current_point_index_band + 1) % len(sorted_points)
                st.rerun()

        if fmin >= fmax:
            st.error("起始频率必须小于终止频率。")
        else:
            # 显示当前测点的结果
            result_df, ref_avg_freq, ref_avg_amp = compare_new_vs_ref_in_band(new_dict, ref_dict, selected_point_band, fmin, fmax)
            st.dataframe(result_df, use_container_width=True)
            
            # 下载按钮 - 生成包含所有测点的完整数据
            all_points_result_df = compare_all_points_in_band(new_dict, ref_dict, sorted_points, fmin, fmax)
            csv_bytes = all_points_result_df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
            st.download_button(
                label="📥 下载差值比较结果 (CSV) - 所有测点",
                data=csv_bytes,
                file_name=f"band_compare_all_points_{fmin:.0f}_{fmax:.0f}Hz.csv",
                mime="text/csv",
                help="下载包含所有测点的完整差值比较结果"
            )

            fig_band = plot_band_overlay(
                new_dict, ref_dict, selected_point_band, fmin, fmax,
                ref_opacity=ref_opacity, ref_line_width=ref_line_width, ref_dash=ref_line_style,
                new_line_width=new_line_width, axis_lower=axis_lower, axis_upper=axis_upper
            )
            st.plotly_chart(fig_band, use_container_width=True)
            st.download_button(
                label="📥 下载频段叠加图 (HTML)",
                data=fig_band.to_html(),
                file_name=f"band_overlay_{selected_point_band}_{fmin:.0f}_{fmax:.0f}Hz.html",
                mime="text/html"
            )

# 返回主页面的导航
st.markdown("---")
st.sidebar.markdown("---")
if st.sidebar.button("🏠 返回主页面"):
    st.switch_page("main_app.py")

st.markdown("---")
st.caption("一致性分析系统 | 基于Streamlit与Plotly | 新数据高亮、参考淡化 | 选定频段差值对比")
