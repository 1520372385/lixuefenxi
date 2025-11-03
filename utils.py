"""
共享工具函数 - 为一致性分析和相似度分析模块提供通用功能
"""

import pandas as pd
import numpy as np
import re

def extract_measurement_points(dfs):
    """
    从数据字典中提取所有测点名称
    """
    points = set()
    for df in dfs.values():
        if df is not None and df.shape[1] >= 2:
            pts = [c for c in df.columns[1:]]
            points.update(pts)
    return points

def sort_measurement_points(points):
    """
    对测点进行排序：按方向(X/Y/Z)和数字排序
    """
    return sorted(
        list(points),
        key=lambda p: (p[:2], int(re.sub(r'\D', '', p) or 0))
    )

def validate_frequency_range(fmin, fmax):
    """
    验证频率范围是否有效
    """
    if fmin >= fmax:
        return False, "起始频率必须小于终止频率"
    return True, ""

def format_point_with_color(point, red_points, yellow_points):
    """
    根据相似度结果格式化测点显示（添加颜色图标）
    """
    if point in red_points:
        return f"🔴 {point}"
    elif point in yellow_points:
        return f"🟡 {point}"
    else:
        return point

def extract_original_point_name(formatted_point):
    """
    从带颜色图标的测点名称中提取原始测点名称
    """
    if formatted_point.startswith("🔴 "):
        return formatted_point[2:]
    elif formatted_point.startswith("🟡 "):
        return formatted_point[2:]
    else:
        return formatted_point

def create_download_button(data, filename, label, mime_type):
    """
    创建通用的下载按钮
    """
    import streamlit as st
    st.download_button(
        label=label,
        data=data,
        file_name=filename,
        mime=mime_type
    )

def create_navigation_buttons(current_index, total_items, session_key):
    """
    创建通用的导航按钮（上一个/下一个）
    """
    import streamlit as st
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("⬅️ 上一个", use_container_width=True):
            if current_index > 0:
                st.session_state[session_key] = current_index - 1
            else:
                st.session_state[session_key] = total_items - 1
            st.rerun()
    
    with col2:
        if st.button("➡️ 下一个", use_container_width=True):
            if current_index < total_items - 1:
                st.session_state[session_key] = current_index + 1
            else:
                st.session_state[session_key] = 0
            st.rerun()

def get_plotly_colors():
    """
    返回Plotly标准颜色列表
    """
    return ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
            '#9467bd', '#8c564b', '#e377c2', '#bcbd22', '#17becf']

def setup_plotly_layout(fig, title, xaxis_title, yaxis_title):
    """
    设置Plotly图表的通用布局
    """
    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        height=600,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode='closest'
    )
    fig.update_xaxes(type="log", gridcolor='lightgray', gridwidth=1, showgrid=True)
    fig.update_yaxes(type="log", gridcolor='lightgray', gridwidth=1, showgrid=True)
    return fig
