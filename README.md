#### 函数功能与调用方式概要 ####

# 数据抽取
python data_extract.py   

# 经纬度拓扑比对
python data_statistic_complain.py plot_geo_coordinates_compare   

# 周内事件统计
pypy data_statistic_complain.py statistic_inwk   
python data_statistic_complain.py plot_statistic_by_time_inwk   

# 事件基本分析（时间、空间、类型分布）
pypy data_statistic_complain.py statistic_week   
python data_statistic_complain.py plot_statistic_by_time_week   
python data_statistic_complain.py plot_statistic_by_space   
python data_statistic_complain.py plot_statistic_simplify   

# 时间、空间相关性分析
pypy analysis_correlation.py generate_tensor   
python analysis_correlation.py plot_time_autocorrelation   
python analysis_correlation.py plot_space_crosscorrelation   

# 区域分析
pypy data_statistic_bycolumn.py statistic_STREETNAME   

# 行业分析
pypy data_statistic_bycolumn.py statistic_EXECUTEDEPTNAME   

# 来源分析
pypy data_statistic_bycolumn.py statistic_INFOSOURCENAME   
pypy data_statistic_bycolumn.py select_important   

# 三林区域分析
python data_statistic_sanlin.py plot_statistic_by_space 部件门牌   

# 聚类分析
python analysis_clustering.py plot_statistic_description_length   
python analysis_clustering.py statistic_description_byevent   
python analysis_clustering.py plot_event_mutual_information   
python -W ignore analysis_clustering.py spatio_temporal_semantic_clustering   

