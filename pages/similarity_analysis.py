"""
相似度分析模块 - 基于yizhixing.py
实现Esim相似度分析，使用参考数据集平均值作为基准向量
包含完整的测点切换和颜色标记功能
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import interpolate
import io
import re
from PIL import Image

# 页面配置
st.set_page_config(
    page_title="相似度分析 - 力学振动数据一致性分析系统",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 应用标题
st.title("📊 相似度分析模块")
st.markdown("Esim相似度分析 - 新数据 vs 参考数据集平均值")
st.markdown("---")

# 转换函数：与一致性分析模块相同的逻辑
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

def format_point_with_location_and_similarity(point_info):
    """格式化测点信息，添加位置信息和相似度值"""
    # 提取测点名称（去掉相似度值）
    if '(' in point_info and ')' in point_info:
        # 格式如: "XM1(0.852)"
        point_name = point_info.split('(')[0]
        similarity_part = point_info[len(point_name):]
    else:
        point_name = point_info
        similarity_part = ""
    
    # 获取位置信息
    location = get_point_location(point_name)
    
    # 格式化输出
    return f"{point_name}({location}){similarity_part}"

def linear_interpolation(data_dict, target_frequencies):
    """
    线性插值函数，将所有数据插值到相同的频率点上
    只进行内插，不外推，范围外填充NaN
    """
    interpolated_data = {}
    
    # 验证目标频率数组
    if target_frequencies is None or len(target_frequencies) == 0:
        st.error("目标频率数组为空或无效")
        return interpolated_data
    
    for label, df in data_dict.items():
        # 获取原始频率和数据
        original_freq = df.iloc[:, 0].values
        
        # 验证原始数据
        if len(original_freq) == 0:
            st.warning(f"数据 '{label}' 的频率数据为空，跳过该数据")
            continue
            
        interpolated_df = pd.DataFrame()
        interpolated_df['HZ'] = target_frequencies
        
        # 对每个测点进行插值
        for col in df.columns[1:]:
            original_data = df[col].values
            
            # 验证测点数据
            if len(original_data) == 0:
                # 如果测点数据为空，填充NaN
                interpolated_df[col] = np.full(len(target_frequencies), np.nan)
                continue
            
            try:
                # 创建插值函数（只内插，不外推）
                f = interpolate.interp1d(original_freq, original_data, 
                                       kind='linear', 
                                       bounds_error=False, 
                                       fill_value=np.nan)  # 范围外填充NaN
                
                # 插值到目标频率点
                interpolated_data_col = f(target_frequencies)
                
                # 确保插值结果长度与目标频率一致
                if len(interpolated_data_col) != len(target_frequencies):
                    # 如果长度不匹配，创建正确长度的数组并填充NaN
                    interpolated_data_col = np.full(len(target_frequencies), np.nan)
                    # 重新插值，确保长度正确
                    try:
                        interpolated_data_col = f(target_frequencies)
                    except:
                        # 如果插值失败，保持NaN数组
                        pass
                
                interpolated_df[col] = interpolated_data_col
                
            except Exception as e:
                # 如果插值失败，填充NaN
                st.warning(f"测点 '{col}' 插值失败: {str(e)}")
                interpolated_df[col] = np.full(len(target_frequencies), np.nan)
        
        interpolated_data[label] = interpolated_df
    
    return interpolated_data

def esim_similarity(x, y, weights=None):
    """
    计算Esim相似度
    公式: s_Esim(X,Y) = 1/n * Σ ω_i * e^(-|x_i-y_i|/(|x_i-y_i|+|x_i+y_i|/2))
    使用等权重1
    """
    n = len(x)
    similarity_sum = 0
    
    # 如果没有提供权重，默认使用等权重
    if weights is None:
        weights = np.ones(n)
    
    for i in range(n):
        diff = abs(x[i] - y[i])
        denominator = diff + abs(x[i] + y[i]) / 2
        if denominator == 0:
            # 避免除零错误，当两者都为0时认为完全相似
            similarity_sum += weights[i]
        else:
            similarity_sum += weights[i] * np.exp(-diff / denominator)
    
    return similarity_sum / n

def calculate_similarity_matrix(new_dict, ref_dict, measurement_points):
    """
    计算相似度矩阵
    X基准向量：参考数据集平均值
    Y向量：新数据测点数据
    """
    # 检查是否有参考数据
    if not ref_dict:
        st.error("没有参考数据，无法计算相似度")
        return None, None
    
    # 自动找到数据点最多的参考数据作为基准
    max_points = 0
    target_frequencies = None
    max_ref_label = None
    
    for label, df in ref_dict.items():
        if df is not None:
            num_points = len(df)
            if num_points > max_points:
                max_points = num_points
                target_frequencies = df.iloc[:, 0].values  # 频率列
                max_ref_label = label
    
    if target_frequencies is None:
        st.error("没有找到有效的参考数据")
        return None, None
    
    st.info(f"使用 '{max_ref_label}' 参考数据的 {max_points} 个频率点作为插值基准")
    
    # 合并新数据和参考数据进行插值
    all_data_dict = {**new_dict, **ref_dict}
    interpolated_data = linear_interpolation(all_data_dict, target_frequencies)
    
    # 计算相似度矩阵
    similarity_matrix = pd.DataFrame(index=list(new_dict.keys()), columns=measurement_points)
    
    # 使用参考数据集平均值作为基准向量
    x_reference = {}
    
    # 计算参考数据集的平均值
    for point in measurement_points:
        # 收集所有参考数据的该测点数据
        all_ref_data = []
        
        for label, df in interpolated_data.items():
            if label in ref_dict and point in df:
                valid_data = df[point].dropna().values
                if len(valid_data) > 0:
                    all_ref_data.append(valid_data)
        
        # 如果收集到了参考数据，计算平均值
        if all_ref_data:
            # 找到最短的数据长度（确保所有向量长度一致）
            min_length = min(len(data) for data in all_ref_data)
            
            # 截取所有数据到相同长度并计算平均值
            trimmed_data = [data[:min_length] for data in all_ref_data]
            average_data = np.mean(trimmed_data, axis=0)
            
            if len(average_data) > 0:
                x_reference[point] = average_data
    
    # 保存基准向量数据到session state
    st.session_state.x_reference = x_reference
    st.session_state.target_frequencies = target_frequencies
    
    # 计算每个新数据文件每个测点的相似度（与参考数据集平均值比较）
    for label in new_dict.keys():
        if label in interpolated_data:
            for point in measurement_points:
                if (point in interpolated_data[label] and point in x_reference):
                    y_data = interpolated_data[label][point].dropna().values
                    x_data = x_reference[point]
                    
                    # 确保向量长度匹配，使用相同长度的数据
                    min_length = min(len(x_data), len(y_data))
                    if min_length > 0:
                        x_trimmed = x_data[:min_length]
                        y_trimmed = y_data[:min_length]
                        
                        # 再次检查长度是否一致
                        if len(x_trimmed) == len(y_trimmed):
                            # 使用等权重1
                            similarity = esim_similarity(x_trimmed, y_trimmed)
                            similarity_matrix.loc[label, point] = similarity
                        else:
                            # 如果长度仍然不匹配，记录NaN
                            similarity_matrix.loc[label, point] = np.nan
                    else:
                        similarity_matrix.loc[label, point] = np.nan
                else:
                    similarity_matrix.loc[label, point] = np.nan
    
    return similarity_matrix, interpolated_data

def plot_similarity_results(similarity_matrix):
    """
    绘制相似度结果图表（X、Y、Z测点分别）
    """
    # 分离X、Y、Z测点
    x_points = [p for p in similarity_matrix.columns if p.startswith('XM')]
    y_points = [p for p in similarity_matrix.columns if p.startswith('YM')]
    z_points = [p for p in similarity_matrix.columns if p.startswith('ZM')]
    
    # 排序测点
    x_points.sort(key=lambda x: int(x[2:]))
    y_points.sort(key=lambda x: int(x[2:]))
    z_points.sort(key=lambda x: int(x[2:]))
    
    # 创建图表
    fig = go.Figure()
    
    # X测点图表
    for file_name in similarity_matrix.index:
        x_values = [int(p[2:]) for p in x_points]
        y_values = [similarity_matrix.loc[file_name, p] for p in x_points]
        fig.add_trace(go.Scatter(
            x=x_values, y=y_values, mode='lines+markers',
            name=f"{file_name} - X测点", line=dict(width=2), marker=dict(size=4)
        ))
    
    # Y测点图表
    for file_name in similarity_matrix.index:
        x_values = [int(p[2:]) for p in y_points]
        y_values = [similarity_matrix.loc[file_name, p] for p in y_points]
        fig.add_trace(go.Scatter(
            x=x_values, y=y_values, mode='lines+markers',
            name=f"{file_name} - Y测点", line=dict(width=2), marker=dict(size=4)
        ))
    
    # Z测点图表
    for file_name in similarity_matrix.index:
        x_values = [int(p[2:]) for p in z_points]
        y_values = [similarity_matrix.loc[file_name, p] for p in z_points]
        fig.add_trace(go.Scatter(
            x=x_values, y=y_values, mode='lines+markers',
            name=f"{file_name} - Z测点", line=dict(width=2), marker=dict(size=4)
        ))
    
    fig.update_layout(
        title="Esim相似度分析结果",
        xaxis_title="测点编号",
        yaxis_title="Esim相似度",
        height=600,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode='closest'
    )
    
    return fig

def find_low_similarity_points(similarity_matrix, threshold=0.8):
    """
    找出相似度低于阈值的测点
    返回格式: {文件名: [测点1, 测点2, ...]}
    """
    low_similarity_results = {}
    
    for file_name in similarity_matrix.index:
        low_points = []
        for point in similarity_matrix.columns:
            similarity_value = similarity_matrix.loc[file_name, point]
            if pd.notna(similarity_value) and similarity_value < threshold:
                low_points.append(f"{point}({similarity_value:.3f})")
        
        if low_points:
            low_similarity_results[file_name] = low_points
    
    return low_similarity_results

def extract_point_names(point_list):
    """
    从测点列表中提取纯测点名称（去掉相似度值）
    例如: "XM1(0.852)" -> "XM1"
    """
    point_names = []
    for point_info in point_list:
        # 提取括号前的测点名称
        if '(' in point_info:
            point_name = point_info.split('(')[0]
            point_names.append(point_name)
        else:
            point_names.append(point_info)
    return point_names

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
    """
    创建频谱图 - 从一致性分析模块移植
    新数据高亮，参考数据淡化
    """
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
if 'current_point_index' not in st.session_state:
    st.session_state.current_point_index = 0
if 'red_points' not in st.session_state:
    st.session_state.red_points = []
if 'yellow_points' not in st.session_state:
    st.session_state.yellow_points = []
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
            for f in new_raw_files:
                df = convert_lms_excel(f)
                if df is not None and df.shape[1] >= 2:
                    st.session_state.new_processed_dict[f.name] = df
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

# 相似度分析
st.subheader("Esim相似度分析（新数据 vs 参考数据集平均值）")
new_dict = st.session_state.new_processed_dict
ref_dict = st.session_state.ref_processed_dict

if not new_dict:
    st.info("请先上传并转换新的未经处理数据。")
elif not ref_dict:
    st.info("请先上传参考数据。")
else:
    new_points = extract_measurement_points(new_dict)
    ref_points = extract_measurement_points(ref_dict)
    common_points = new_points & ref_points

    if not common_points:
        st.warning("未找到可在新与参考之间共同叠加的测点。")
    else:
        sorted_points = sorted(
            list(common_points),
            key=lambda p: (p[:2], int(re.sub(r'\D', '', p) or 0))
        )
        
        # 测点选择区域
        cols = st.columns([3, 1, 1])
        with cols[0]:
            # 创建带颜色标记和位置信息的测点选项
            def format_point_with_color_and_location(point):
                location = get_point_location(point) if st.session_state.location_data else ""
                location_suffix = f" ({location})" if location else ""
                
                if point in st.session_state.red_points:
                    return f"🔴 {point}{location_suffix}"
                elif point in st.session_state.yellow_points:
                    return f"🟡 {point}{location_suffix}"
                else:
                    return f"{point}{location_suffix}"
            
            if st.session_state.location_data:
                point_options = [format_point_with_color_and_location(point) for point in sorted_points]
            else:
                point_options = [format_point_with_color_and_location(point) for point in sorted_points]
            
            selected_point_with_location = st.selectbox(
                "选择测点",
                point_options,
                index=min(st.session_state.current_point_index, len(point_options)-1)
            )
        
        # 提取原始测点名称（去掉颜色标记和位置信息）
        if st.session_state.location_data:
            # 从带颜色标记和位置信息的选项中提取原始测点名称
            selected_point = selected_point_with_location.split(' (')[0].replace('🔴 ', '').replace('🟡 ', '')
        else:
            selected_point = selected_point_with_location.replace('🔴 ', '').replace('🟡 ', '')
        with cols[1]:
            if st.button("⬅️ 上一个", use_container_width=True):
                st.session_state.current_point_index = (st.session_state.current_point_index - 1) % len(sorted_points)
                st.rerun()
        with cols[2]:
            if st.button("➡️ 下一个", use_container_width=True):
                st.session_state.current_point_index = (st.session_state.current_point_index + 1) % len(sorted_points)
                st.rerun()

        # 创建带颜色标记的下拉菜单选项
        def format_point_option(point):
            if point in st.session_state.red_points:
                return f"🔴 {point}"
            elif point in st.session_state.yellow_points:
                return f"🟡 {point}"
            else:
                return point

        formatted_points = [format_point_option(point) for point in sorted_points]
        
        # 显示当前测点信息
        st.info(f"当前测点: {selected_point} ({st.session_state.current_point_index + 1}/{len(sorted_points)})")

        # 频谱分析显示
        st.subheader("📊 频谱分析")
        spectrum_fig = create_spectrum_plot_emphasis(
            new_dict, ref_dict, selected_point,
            ref_opacity=0.75, ref_line_width=1.6, ref_dash='dash',
            new_line_width=2.5
        )
        st.plotly_chart(spectrum_fig, use_container_width=True)

        # 清除颜色标记按钮
        if st.button("🗑️ 清除所有颜色标记", use_container_width=True):
            st.session_state.red_points = []
            st.session_state.yellow_points = []
            st.rerun()

        # 自动进行相似度分析
        if 'similarity_matrix' not in st.session_state or st.session_state.get('current_files') != (tuple(new_dict.keys()), tuple(ref_dict.keys())):
            with st.spinner("正在进行相似度分析..."):
                try:
                    st.session_state.similarity_matrix, st.session_state.interpolated_data = calculate_similarity_matrix(
                        new_dict, ref_dict, sorted_points
                    )
                    st.session_state.similarity_calculated = True
                    st.session_state.current_files = (tuple(new_dict.keys()), tuple(ref_dict.keys()))
                except Exception as e:
                    st.error(f"相似度分析失败: {str(e)}")
                    st.session_state.similarity_calculated = False
        else:
            st.session_state.similarity_calculated = True

        # 显示相似度分析结果
        if st.session_state.similarity_calculated and 'similarity_matrix' in st.session_state:
            # 主界面显示详细结果
            st.subheader("📊 相似度结果表格")
            st.dataframe(st.session_state.similarity_matrix, use_container_width=True)

            # 绘制相似度图表
            st.subheader("📈 相似度分析图表")
            fig = plot_similarity_results(st.session_state.similarity_matrix)
            st.plotly_chart(fig, use_container_width=True)

            # 添加下载按钮（表格数据）
            csv_data = st.session_state.similarity_matrix.to_csv().encode('utf-8')
            st.download_button(
                label="📥 下载CSV格式数据",
                data=csv_data,
                file_name="similarity_results.csv",
                mime="text/csv"
            )


            # 显示严重低相似度结果框 (阈值 0.8)
            st.subheader("⚠️ 严重低相似度测点识别 (相似度 < 0.8)")
            low_similarity_results_08 = find_low_similarity_points(st.session_state.similarity_matrix, threshold=0.8)

            if low_similarity_results_08:
                # 提取所有<0.8的测点名称
                all_low_points_08 = []
                for low_points in low_similarity_results_08.values():
                    point_names = extract_point_names(low_points)
                    all_low_points_08.extend(point_names)

                # 添加标红按钮
                if st.button("🔴 标红所有<0.8测点", key="mark_red_08"):
                    st.session_state.red_points = list(set(st.session_state.red_points + all_low_points_08))
                    st.rerun()

                for file_name, low_points in low_similarity_results_08.items():
                    with st.expander(f"❌ {file_name} - 严重低相似度测点 ({len(low_points)}个)"):
                        st.markdown("**测点列表 (相似度值):**")
                        for point_info in low_points:
                            # 添加位置信息显示
                            point_with_location = format_point_with_location_and_similarity(point_info)
                            st.markdown(f"- {point_with_location}")
            else:
                st.success("🎉 所有测点的相似度均 ≥ 0.8，结果优秀！")

            # 显示中等低相似度结果框 (阈值 0.8~0.9)
            st.subheader("⚠️ 中等低相似度测点识别 (相似度在 0.8~0.9)")

            # 找出相似度在0.8~0.9之间的测点
            medium_low_similarity_results = {}

            for file_name in st.session_state.similarity_matrix.index:
                medium_low_points = []
                for point in st.session_state.similarity_matrix.columns:
                    similarity_value = st.session_state.similarity_matrix.loc[file_name, point]
                    if pd.notna(similarity_value) and 0.8 <= similarity_value < 0.9:
                        medium_low_points.append(f"{point}({similarity_value:.3f})")

                if medium_low_points:
                    medium_low_similarity_results[file_name] = medium_low_points

            if medium_low_similarity_results:
                # 提取所有0.8~0.9的测点名称
                all_medium_low_points = []
                for medium_low_points in medium_low_similarity_results.values():
                    point_names = extract_point_names(medium_low_points)
                    all_medium_low_points.extend(point_names)

                # 添加标黄按钮
                if st.button("🟡 标黄所有0.8~0.9测点", key="mark_yellow_08_09"):
                    st.session_state.yellow_points = list(set(st.session_state.yellow_points + all_medium_low_points))
                    st.rerun()

                for file_name, medium_low_points in medium_low_similarity_results.items():
                    with st.expander(f"📋 {file_name} - 中等低相似度测点 ({len(medium_low_points)}个)"):
                        st.markdown("**测点列表 (相似度值):**")
                        for point_info in medium_low_points:
                            # 添加位置信息显示
                            point_with_location = format_point_with_location_and_similarity(point_info)
                            st.markdown(f"- {point_with_location}")
            else:
                st.success("🎉 所有测点的相似度均 ≥ 0.9 或 < 0.8，结果良好！")

            st.success("✅ 相似度分析已完成！")
        else:
            st.warning("请至少上传一个有效的新数据和参考数据")

# 返回主页面的导航
st.markdown("---")
st.sidebar.markdown("---")
if st.sidebar.button("🏠 返回主页面"):
    st.switch_page("main_app.py")

# 页脚信息
st.markdown("---")
st.caption("相似度分析系统 | 基于Streamlit和Plotly开发 | 使用参考数据集平均值作为基准向量")

# 在模块最后显示Esim公式说明
st.markdown("---")
st.subheader("📐 Esim相似度公式说明")

# 显示公式图片
try:
    formula_image = Image.open('esim_formula.png')
    st.image(formula_image, caption="Esim相似度计算公式", use_container_width=True)
except FileNotFoundError:
    st.warning("Esim公式图片未找到，请确保esim_formula.png文件存在")

# 显示公式说明
st.markdown("""
**公式说明：**

- **X**: 参考数据集平均值向量（所有参考数据在某测点的平均值）
- **Y**: 新数据测点数据向量（新数据文件在某测点的数据）
- **相似度**: 基于Esim公式计算的相似度值，范围在0-1之间
- **n**: 频率点数
- **ω_i**: 权重系数（当前使用等权重1）
- **x_i, y_i**: 第i个频率点的幅值

**公式特点：**
- 使用指数衰减函数度量相似度
- 考虑了绝对差异和平均值的相对关系
- 当X=Y时，相似度=1（完全相似）
- 当X和Y差异很大时，相似度趋近于0
""")
