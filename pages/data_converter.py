"""
LMS数据转换模块 - 集成到主应用
专门用于将LMS导出的Excel文件转换为简洁的数据格式
"""

import streamlit as st
import pandas as pd
import io
import re
import zipfile

# 页面配置
st.set_page_config(
    page_title="LMS数据转换 - 力学振动数据一致性分析系统",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 应用标题
st.title("🔄 LMS数据转换工具")
st.markdown("将LMS导出的原始Excel文件转换为简洁的特征级数据格式")
st.markdown("---")

def process_lms_excel(file):
    """
    处理LMS导出的Excel文件，提取有效数据（仅保留频率和第12行偶数列测点响应）
    """
    try:
        excel_file = pd.ExcelFile(file)
        sheet_names = excel_file.sheet_names
        st.info(f"检测到 {len(sheet_names)} 个工作表: {', '.join(sheet_names)}")
        # 读取原始数据（不设header）
        df = pd.read_excel(file, sheet_name=sheet_names[0], header=None)
        # 获取第12行（Excel第12行，pandas索引为11）
        row12 = df.iloc[11]
        # 偶数列（Excel第2、4、6...列，pandas索引为1,3,5...）
        measurement_indices = [i for i in range(1, len(row12), 2) if pd.notna(row12[i])]
        measurement_names = [str(row12[i]).strip() for i in measurement_indices]
        # 频率列始终为第1列（pandas索引0）
        frequency_col = 0
        # 数据起始行（假设数据从第13行开始，即pandas索引12）
        data_start_row = 12
        df_data = df.iloc[data_start_row:]
        # 构建新DataFrame
        processed_df = pd.DataFrame()
        processed_df['HZ'] = df_data[frequency_col].reset_index(drop=True)
        # 只保留测点名称中的XM/YM/ZM等具体信息，去除'Peak Spectrum'前缀
        for idx, name in zip(measurement_indices, measurement_names):
            # 提取测点名（如XM1、YM2等）
            match = re.search(r'(XM\d+|YM\d+|ZM\d+)', name)
            if match:
                point_name = match.group(1)
            else:
                point_name = name  # 如果没有匹配则保留原名
            processed_df[point_name] = df_data[idx].reset_index(drop=True)
        # 清理数据（去除空值和无效数据）
        processed_df = processed_df.dropna()
        processed_df['HZ'] = pd.to_numeric(processed_df['HZ'], errors='coerce')
        for col in processed_df.columns[1:]:
            processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
        processed_df = processed_df.dropna()
        return processed_df
    except Exception as e:
        st.error(f"文件处理出错: {str(e)}")
        return None

# 功能介绍
st.markdown("""
### 🎯 功能说明

- **输入**: LMS导出的原始Excel文件（包含大量元数据）
- **输出**: 简洁的数据格式（仅保留频率和测点响应数据）
- **支持**: XM、YM、ZM等测点格式
- **批量处理**: 支持同时处理多个文件
""")

# 文件上传区域
st.header("📁 批量文件上传")
uploaded_files = st.file_uploader(
    "上传多个LMS导出的Excel文件（可批量）",
    type=['xlsx', 'xls'],
    accept_multiple_files=True,
    help="支持批量上传.xlsx和.xls格式的LMS导出文件"
)

if uploaded_files:
    st.success(f"✅ 成功上传 {len(uploaded_files)} 个文件")
    results = []
    
    for uploaded_file in uploaded_files:
        with st.spinner(f"正在处理文件: {uploaded_file.name}..."):
            processed_df = process_lms_excel(uploaded_file)
        
        if processed_df is not None:
            st.header(f"📊 处理结果 - {uploaded_file.name}")
            
            # 显示数据预览
            col1, col2 = st.columns([2, 1])
            with col1:
                st.dataframe(processed_df.head(10), use_container_width=True)
            with col2:
                st.metric("数据行数", len(processed_df))
                st.metric("测点数量", len(processed_df.columns) - 1)
            
            # 创建Excel文件
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                processed_df.to_excel(writer, sheet_name='转换数据', index=False)
            excel_buffer.seek(0)
            
            # 下载按钮
            st.download_button(
                label=f"📥 下载转换后的Excel文件 - {uploaded_file.name}",
                data=excel_buffer,
                file_name=f"converted_{uploaded_file.name}",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                help="点击下载处理后的Excel文件",
                use_container_width=True
            )
            results.append((uploaded_file.name, excel_buffer.getvalue()))
    
    # 批量打包下载
    if results:
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zipf:
            for fname, fdata in results:
                zipf.writestr(f"converted_{fname}", fdata)
        zip_buffer.seek(0)
        
        st.download_button(
            label="📦 下载全部转换结果（ZIP打包）",
            data=zip_buffer,
            file_name="converted_results.zip",
            mime="application/zip",
            help="下载所有转换后的Excel文件打包ZIP",
            use_container_width=True
        )
    
    st.success("✅ 所有文件处理完成！可分别或批量下载转换结果。")
else:
    st.info("👆 请在上方上传LMS导出的Excel文件（可批量上传）")
    
    # 使用示例
    with st.expander("📖 使用示例"):
        st.markdown("""
        ### 输入文件格式 (原始LMS导出)
        ```
        ... (大量元数据) ...
        Hz    g
        5     0.1415315424911
        5.119 0.1483887707059
        5.219 0.1511065040076
        ... (更多数据) ...
        ```
        
        ### 输出文件格式 (转换后)
        ```
        HZ    XM1            XM2            XM3
        5     0.1415315424911 0.1438960720875 0.1441654239495
        5.119 0.1483887707059 0.1505708271187 0.1510683829029
        5.219 0.1511065040076 0.1514430418856 0.1505309312431
        ... (更多数据) ...
        ```
        
        ### 转换规则
        - 提取第12行作为测点名称
        - 保留偶数列作为测点数据
        - 第一列作为频率数据
        - 从第13行开始提取有效数据
        - 自动清理无效值和空值
        """)

# 返回主页面的导航
st.markdown("---")
st.sidebar.markdown("---")
if st.sidebar.button("🏠 返回主页面"):
    st.switch_page("main_app.py")

# 页脚信息
st.markdown("---")
st.caption("LMS数据转换工具 | 基于Streamlit开发 | 集成到力学振动数据一致性分析系统")
