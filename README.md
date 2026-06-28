# 超市五店销售数据分析项目
## 项目简介
基于2018-2022门店销售明细，完成数据清洗、EDA探索、KMeans波士顿矩阵品类聚类、Prophet时序销售额预测、Streamlit可视化交互式仪表盘。
## 文件说明
1. main_analysis.py：数据预处理、季度/月度时序分析、品类聚类、销售预测主程序
2. dashboard.py：Streamlit网页可视化仪表盘，支持门店/品类/年份筛选
3. requirements.txt：项目全部Python依赖库
## 运行方法
1. 安装依赖：pip install -r requirements.txt
2. 执行数据分析：python main_analysis.py
3. 启动可视化仪表盘：streamlit run dashboard.py
## 输出内容
静态分析图表、汇总Excel经营报表、交互式网页看板
## 项目仓库地址
https://github.com/chen-shu-sen/supermarket_sales_analysis