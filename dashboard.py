# 超市销售交互式仪表盘 Streamlit
# 运行命令：streamlit run dashboard.py
import streamlit as st
import pandas as pd
import plotly.express
import warnings
warnings.filterwarnings("ignore")

# 页面全局配置
st.set_page_config(page_title="超市五店销售数据仪表盘", layout="wide")
# 隐藏侧边栏水印、页脚、头部（修复三引号多行字符串语法错误）
hide_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;
</style>
"""
st.markdown(hide_style, unsafe_allow_html=True)

# 缓存读取清洗后数据
@st.cache_data
def load_data():
    df = pd.read_excel(r"C:/Users/陈/Desktop/2/超市五店销售数据.xlsx", sheet_name="清洗后明细数据")
    df["年月"] = df["年月"].astype(str)
    return df

df = load_data()

# 侧边栏筛选控件
st.sidebar.header("数据筛选面板")
store_filter = st.sidebar.multiselect("选择门店", df["门店名称"].unique(), default=df["门店名称"].unique())
year_filter = st.sidebar.multiselect("选择年份", sorted(df["年"].unique()), default=sorted(df["年"].unique()))
quarter_filter = st.sidebar.multiselect("选择季度", sorted(df["季度"].unique()), default=sorted(df["季度"]))
category_filter = st.sidebar.multiselect("产品类别", df["产品类别"].unique(), default=df["产品类别"][:10])

# 筛选后数据
df_filter = df[
    (df["门店名称"].isin(store_filter)) &
    (df["年"].isin(year_filter)) &
    (df["季度"].isin(quarter_filter)) &
    (df["产品类别"].isin(category_filter))
]

# 页面标题
st.title("超市五店经营数据可视化仪表盘")
st.divider()

# 第一行：核心KPI卡片（修复sum()缺少括号bug）
col1, col2, col3, col4 = st.columns(4)
total_sales = df_filter["销售额"].sum()
total_profit = df_filter["利润额"].sum()
total_order = df_filter["订单量"].sum()
avg_profit_rate = total_profit / total_sales if total_sales != 0 else 0

with col1:
    st.metric("总销售额", f"{total_sales:,.0f} 元")
with col2:
    st.metric("总利润", f"{total_profit:,.0f} 元")
with col3:
    st.metric("总订单量", f"{total_order:,} 单")
with col4:
    st.metric("整体利润率", f"{avg_profit_rate:.2%}")

st.divider()

# 第二行：月度趋势图
col_a, col_b = st.columns(2)
with col_a:
    monthly_agg = df_filter.groupby("年月")[["销售额", "利润额"]].sum().reset_index()
    fig_line = plotly.express.line(monthly_agg, x="年月", y=["销售额", "利润额"], title="月度销售&利润趋势")
    st.plotly_chart(fig_line, use_container_width=True)
with col_b:
    store_agg = df_filter.groupby("门店名称")[["销售额", "利润额"]].sum().reset_index()
    fig_bar = plotly.express.bar(store_agg, x="门店名称", y="销售额", color="门店名称", title="各门店销售额对比")
    st.plotly_chart(fig_bar, use_container_width=True)

# 第三行：品类、订单等级分析（修复不存在的订单类别字段）
col_c, col_d = st.columns(2)
with col_c:
    cat_agg = df_filter.groupby("产品类别")["销售额"].sum().sort_values(ascending=False).head(10).reset_index()
    fig_pie = plotly.express.pie(cat_agg, values="销售额", names="产品类别", title="TOP10品类销售额占比")
    st.plotly_chart(fig_pie, use_container_width=True)
with col_d:
    fig_box = plotly.express.box(df_filter, x="订单等级", y="利润率", title="各订单等级利润率分布")
    st.plotly_chart(fig_box, use_container_width=True)

# 底部明细数据
st.divider()
st.subheader("筛选后原始明细数据")
st.dataframe(df_filter, height=350)