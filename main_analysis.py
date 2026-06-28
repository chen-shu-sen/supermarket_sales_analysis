# ====================== 导入依赖库 ======================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from prophet import Prophet
from scipy.stats import f_oneway, kruskal
import warnings
# 仅屏蔽无关警告，不掩盖报错
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# 设置matplotlib中文显示（解决图表方块乱码）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ====================== 1 数据读取与预处理 ======================
# 1.1 读取原始数据
def load_raw_data(file_path: str, sheet_name: str):
    df_raw = pd.read_excel(file_path, sheet_name=sheet_name)
    print("========== 原始数据基本信息 ==========")
    print(f"数据总行数：{df_raw.shape[0]}，总字段数：{df_raw.shape[1]}")
    print("\n字段类型：")
    print(df_raw.info())
    print("\n缺失值统计：")
    print(df_raw.isnull().sum())
    return df_raw

df_raw = load_raw_data("42705.xlsx", "某超市公司五店销售数据")

# 1.2 基础清洗：空值、无效订单剔除
# 删除订单量/销售额/利润为空；剔除订单量=0无效单
df_raw.dropna(subset=["订单量", "销售额", "利润额"], inplace=True)
df_raw = df_raw[df_raw["订单量"] > 0]

# 1.3 日期字段衍生
df_raw["订单日期"] = pd.to_datetime(df_raw["订单日期"])
df_raw["年"] = df_raw["订单日期"].dt.year
df_raw["月"] = df_raw["订单日期"].dt.month
df_raw["季度"] = df_raw["订单日期"].dt.quarter
df_raw["年月"] = df_raw["订单日期"].dt.to_period("M")

# 1.4 核心衍生指标：利润率（规避销售额=0除零报错）
df_raw["利润率"] = np.where(
    df_raw["销售额"] == 0,
    np.nan,
    df_raw["利润额"] / df_raw["销售额"]
)
df_raw["是否负利润"] = df_raw["利润额"] < 0

# 1.5 异常值过滤：业务阈值优先，再3σ过滤
# 业务常识：利润率-1 ~ 1 之外为录入错误
df_step1 = df_raw[(df_raw["利润率"] > -1) & (df_raw["利润率"] < 1)]
# 3倍标准差剔除极端异常利润率
profit_mean = df_step1["利润率"].mean()
profit_std = df_step1["利润率"].std
df_clean = df_step1[(df_step1["利润率"] >= profit_mean - 3 * profit_std) &
                    (df_step1["利润率"] <= profit_mean + 3 * profit_std)]
print(f"\n清洗前数据量：{df_raw.shape[0]}，清洗后有效数据量：{df_clean.shape[0]}")

# ====================== 2 探索性数据分析 EDA ======================
# 2.1 时间维度聚合：月度、季度销售利润
## 月度聚合
monthly = df_clean.groupby("年月")[["销售额", "利润额"]].sum().reset_index()
monthly["年月_str"] = monthly["年月"].astype(str)

# 绘制月度趋势图
plt.figure(figsize=(14, 6))
plt.plot(monthly["年月_str"], monthly["销售额"], linewidth=2, label="销售额")
plt.plot(monthly["年月"], monthly["利润额"], linewidth=2, label="利润额")
plt.xticks(rotation=45, fontsize=9)
plt.xlabel("年月")
plt.ylabel("金额")
plt.title("2018-2022 月度销售额&利润趋势")
plt.legend()
plt.tight_layout()
plt.savefig("月度销售利润趋势.png", dpi=300)
plt.show()

## 季度聚合
quarterly = df_clean.groupby(["年", "季度"])[["销售额", "利润额"]].sum().reset_index()
quarterly["年季"] = quarterly["年"].astype(str) + "-Q" + quarterly["季度"].astype(str)
print("\n===== 季度汇总数据 =====")
print(quarterly)

# 2.2 产品品类分析：销售额占比、加权利润率
# 销售额总额排序
category_sales = df_clean.groupby("产品类别")["销售额"].sum().sort_values(ascending=False)
sales_ratio = category_sales / category_sales.sum() * 100

# 加权平均利润率（总利润/总销售额）
category_profit = df_clean.groupby("产品类别").apply(
    lambda x: x["利润额"].sum() / x["销售额"].sum()
).sort_values(ascending=False)

# 品类分析总表
category_df = pd.DataFrame({
    "总销售额": category_sales,
    "销售额占比(%)": sales_ratio,
    "加权利润率": category_profit
})
print("\n===== 全品类经营分析 =====")
print(category_df.round(4))

# 2.3 订单等级利润率分布 + 显著性检验
# 箱线图
plt.figure(figsize=(10, 6))
sns.boxplot(x="订单等级", y="利润率", data=df_clean)
plt.title("不同订单等级利润率分布")
plt.savefig("订单等级利润率箱线图.png", dpi=300)
plt.show()

# ANOVA检验（先正态/方差前提不足，补充非参数检验）
level_list = df_clean["订单等级"].unique()
group_data = [df_clean[df_clean["订单等级"] == lv]["利润率"] for lv in level_list]
# 参数检验ANOVA
f_stat, p_anova = f_oneway(*group_data)
# 非参数克鲁斯卡尔检验（稳健结果）
stat_k, p_krus = kruskal(*group_data)
print("\n===== 订单等级利润率差异检验 =====")
print(f"ANOVA F统计量：{round(f_stat,4)}，P值：{round(p_anova,6)}")
print(f"Kruskal非参数检验P值：{round(p_krus,6)}")
if p_krus < 0.05:
    print("结论：不同订单等级利润率存在显著差异")
else:
    print("结论：不同订单等级利润率无显著差异")

# 2.4 门店维度整体KPI
store_kpi = df_clean.groupby("门店名称").agg({
    "销售额": "sum",
    "利润额": "sum",
    "订单量": "sum",
    "利润率": lambda x: (x["利润额"].sum() / x["销售额"].sum())
}).rename(columns={"利润率": "整体利润率"})
print("\n===== 各门店经营KPI =====")
print(store_kpi.round(4))

# ====================== 3 KMeans聚类（波士顿矩阵四象限） ======================
# 3.1 构建聚类数据集：各品类总销售额、平均利润率
cluster_raw = df_clean.groupby("产品类别").agg({
    "销售额": "sum",
    "利润率": "mean"
}).reset_index()

# 3.2 肘部法则+轮廓系数 确定最优聚类k
scaler = StandardScaler()
X_scaled = scaler.fit_transform(cluster_raw[["销售额", "利润率"]])
sse = []
sil_score_list = []
k_range = range(2, 8)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42)
    label = km.fit_predict(X_scaled)
    sse.append(km.inertia_)
    sil = silhouette_score(X_scaled, label)
    sil_score_list.append(sil)

# 绘制肘部图
plt.figure(figsize=(10, 4))
plt.subplot(1, 2)
plt.plot(k_range, sse, marker="o")
plt.xlabel("聚类数量k")
plt.ylabel("SSE误差平方和")
plt.title("肘部法则确定最优k值")
# 轮廓系数图
plt.subplot(1, 2)
plt.plot(k_range, sil_score_list, marker="o", color="orange")
plt.xlabel("聚类数量k")
plt.ylabel("轮廓系数")
plt.title("轮廓系数（越高聚类效果越好）")
plt.tight_layout()
plt.savefig("聚类k值筛选.png", dpi=300)
plt.show()

# 业务固定k=4（波士顿矩阵四类）
kmeans_model = KMeans(n_clusters=4, random_state=42)
cluster_raw["cluster"] = kmeans_model.fit_predict(X_scaled)

# 输出每类均值，匹配波士顿矩阵
cluster_summary = cluster_raw.groupby("cluster")[["销售额", "利润率"]].mean()
print("\n===== 四类聚类指标均值 =====")
print(cluster_summary.round(2))

# 绘制品类聚类散点图
plt.figure(figsize=(12, 7))
sns.scatterplot(
    data=cluster_raw,
    x="销售额", y="利润率",
    hue="cluster", size="销售额", sizes=(40, 350),
    palette="Set2"
)
plt.title("品类聚类波士顿矩阵（销售额-利润率四象限）")
plt.savefig("品类聚类散点图.png", dpi=300)
plt.show()

# 给聚类手动标注业务类型（根据均值匹配）
def label_cluster(row):
    c = row["cluster"]
    sale = cluster_summary.loc[c, "销售额"]
    profit = cluster_summary.loc[c, "利润率"]
    if sale > cluster_summary["销售额"].mean() and profit > cluster_summary["利润率"].mean():
        return "明星品类（高销高利）"
    elif sale > cluster_summary["销售额"] and profit < cluster_summary["利润率"].mean():
        return "问题品类（高销低利）"
    elif sale < cluster_summary["销售额"] and profit > cluster_summary["利润率"].mean():
        return "金牛品类（低销高利）"
    else:
        return "瘦狗品类（低销低利）"

cluster_raw["品类类型"] = cluster_raw.apply(label_cluster, axis=1)
print("\n===== 全部品类波士顿矩阵分类结果 =====")
print(cluster_raw[["产品类别", "总销售额", "平均利润率", "cluster", "品类类型"]])

# ====================== 4 Prophet销售额时序预测 ======================
# 构造Prophet标准格式 ds时间 y目标值
prop_data = monthly.copy()
prop_data["ds"] = pd.to_datetime(prop_data["年月_str"])
prop_data["y"] = prop_data["销售额"]
prop_model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=False,
    daily_seasonality=False,
    seasonality_prior_scale=10
)
prop_model.fit(prop_data)

# 预测未来12个月
future_df = prop_model.make_future_dataframe(periods=12, freq="M")
forecast_result = prop_model.predict(future_df)

# 绘制预测图
fig1 = prop_model.plot(forecast_result)
plt.title("月度销售额历史+未来12个月预测")
plt.savefig("销售额预测图.png", dpi=300)
plt.show()

# 预测趋势分解图（趋势+年度季节）
fig2 = prop_model.plot_components(forecast_result)
plt.savefig("预测趋势分解.png", dpi=300)
plt.show()

# 输出未来12个月预测明细
pred_future = forecast_result.tail(12)[["ds", "yhat", "yhat_lower", "yhat_upper"]]
pred_future.columns = ["预测年月", "预测销售额", "下限", "上限"]
print("\n===== 未来12个月销售额预测 =====")
print(pred_future.round(0))

# ====================== 5 数据导出（用于论文附表） ======================
# 清洗后明细、品类聚类、门店KPI、预测结果导出Excel
with pd.ExcelWriter("超市分析结果汇总.xlsx") as writer:
    df_clean.to_excel(writer, sheet_name="清洗明细", index=False)
    store_kpi.to_excel(writer, sheet_name="门店KPI")
    category_df.to_excel(writer, sheet_name="品类经营")
    cluster_raw.to_excel(writer, sheet_name="波士顿聚类")
    pred_future.to_excel(writer, sheet_name="销售预测")
    quarterly.to_excel(writer, sheet_name="季度汇总")
print("\n全部分析结果已导出至：超市分析结果汇总.xlsx")