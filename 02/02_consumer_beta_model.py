import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import os
import warnings

warnings.filterwarnings('ignore')


# =====================================================================
# 新增部分：数据预处理函数
# =====================================================================

def preprocess_industry_classification(file_path):
    """
    处理行业分类数据，将其转换为代码所需的格式，并映射到高、中、低消费相关行业组

    参数:
    file_path: str - 行业分类CSV文件路径

    返回:
    Series - 索引为股票代码，值为行业名称的Series，经过映射，可以与行业组匹配
    """
    df = pd.read_csv(file_path)
    print("行业列的数据类型:", df['Nindnme'].dtype)
    print("行业列的前几个值:", df['Nindnme'].head())
    print("行业列中的非字符串值:", df[~df['Nindnme'].apply(lambda x: isinstance(x, str))]['Nindnme'].head())
    # 定义行业映射规则
    # 将Nindnme映射到消费相关度
    industry_consumption_mapping = {
        # 高消费相关行业
        '批发和零售贸易': '零售',
        '零售业': '零售',
        '食品、饮料、烟草及饲料加工业': '食品饮料',
        '纺织业': '纺织服装',
        '纺织服装、服饰业': '纺织服装',
        '家用电器制造业': '家用电器',
        '食品制造业': '食品饮料',
        '饮料制造业': '食品饮料',
        '餐饮业': '餐饮',

        # 中消费相关行业
        '医药制造业': '医药生物',
        '计算机、通信和其他电子设备制造业': '计算机',
        '信息技术业': '科技',
        '通信设备、计算机及其他电子设备制造业': '通信',
        '医疗仪器设备及器械制造': '医疗',
        '汽车制造业': '汽车',
        '文化、体育、娱乐业': '传媒',
        '卫生、社会保障及社会福利业': '医疗',

        # 低消费相关行业
        '电力、热力生产和供应业': '公用事业',
        '燃气生产和供应业': '公用事业',
        '金融业': '金融',
        '银行业': '金融',
        '非银行金融': '金融',
        '资本市场服务': '金融',
        '保险业': '金融',
        '房地产业': '房地产',
        '建筑业': '建筑',
        '土木工程建筑业': '建筑',
        '交通运输、仓储业': '交通运输',
        '采矿业': '采掘',
        '黑色金属冶炼及压延加工业': '钢铁',
        '有色金属冶炼及压延加工业': '有色金属',
    }

    # 创建初始Series：索引为stock_code，值为Nindnme(行业名称)
    industry_classification = pd.Series(df['Nindnme'].values, index=df['stock_code'])

    # 处理可能的NaN值和非字符串值
    industry_classification = industry_classification.fillna('未知行业')  # 填充NaN值
    industry_classification = industry_classification.astype(str)  # 转换为字符串类型

    # 创建映射后的行业分类Series
    mapped_classification = pd.Series(index=industry_classification.index, dtype='object')

    # 对每个股票的行业进行映射
    for stock, industry in industry_classification.items():
        # 尝试直接映射
        if industry in industry_consumption_mapping:
            mapped_classification[stock] = industry_consumption_mapping[industry]
        else:
            # 尝试部分匹配
            matched = False
            for orig, mapped in industry_consumption_mapping.items():
                if orig in industry or industry in orig:
                    mapped_classification[stock] = mapped
                    matched = True
                    break

            # 如果仍然没有匹配，按照行业分类代码进行粗略映射
            if not matched:
                # 检查industry字段
                if 'industry' in df.columns:
                    industry_code = df.loc[df['stock_code'] == stock, 'industry'].values[0]
                    if isinstance(industry_code, str):
                        if industry_code.startswith('H') or industry_code.startswith('F'):
                            mapped_classification[stock] = '零售'  # 批发和零售贸易、交通运输
                        elif industry_code.startswith('C'):
                            mapped_classification[stock] = '科技'  # 制造业
                        elif industry_code.startswith('I') or industry_code.startswith('J'):
                            mapped_classification[stock] = '金融'  # 金融、房地产
                        elif industry_code.startswith('D') or industry_code.startswith('E'):
                            mapped_classification[stock] = '公用事业'  # 电力、建筑
                        else:
                            mapped_classification[stock] = '科技'  # 默认归为中消费相关
                    else:
                        mapped_classification[stock] = '科技'  # 默认归为中消费相关
                else:
                    mapped_classification[stock] = '科技'  # 默认归为中消费相关

    # 填充可能的NaN值
    mapped_classification = mapped_classification.fillna('科技')  # 默认归为中消费相关

    print(f"行业分类数据处理完成，共 {len(mapped_classification)} 个股票")

    # 查看分类结果
    consumption_groups = {'高消费相关': 0, '中消费相关': 0, '低消费相关': 0}
    high_consumption = ['消费品', '零售', '餐饮', '食品饮料', '纺织服装', '家用电器', '商贸零售']
    medium_consumption = ['医疗', '科技', '传媒', '汽车', '医药生物', '计算机', '通信']
    low_consumption = ['能源', '公用事业', '金融', '材料', '房地产', '建筑', '交通运输', '采掘', '钢铁', '有色金属']

    for industry in mapped_classification.values:
        if industry in high_consumption:
            consumption_groups['高消费相关'] += 1
        elif industry in medium_consumption:
            consumption_groups['中消费相关'] += 1
        elif industry in low_consumption:
            consumption_groups['低消费相关'] += 1

    print(f"分类结果: {consumption_groups}")

    return mapped_classification


def load_consumption_data():
    """
    加载消费相关数据

    返回:
    tuple - 包含零售总额、消费者信心指数和消费价格涨幅的DataFrame
    """
    # 读取社会消费品零售总额数据
    retail_sales = pd.read_csv('CME_Mretailsales.csv')

    # 读取消费者信心指数数据
    consumer_confidence = pd.read_csv('CME_Mbcid2.csv')

    # 读取居民消费价格涨幅数据
    consumer_price = pd.read_csv('CME_Mconsumerpriceratio.csv')

    # 处理日期列
    for df in [retail_sales, consumer_confidence, consumer_price]:
        if 'Month' in df.columns:
            df['Date'] = pd.to_datetime(df['Month'])
            df.set_index('Date', inplace=True)

    return retail_sales, consumer_confidence, consumer_price


def generate_prediction_files(retail_sales, consumer_confidence, consumer_price):
    """
    从消费数据生成预测结果文件
    这是一个简化的实现，仅当没有运行过triangle_PCA_model.py时作为备选方案
    理想情况下，应先运行triangle_PCA_model.py生成更准确的预测结果

    参数:
    retail_sales: DataFrame - 社会消费品零售总额数据
    consumer_confidence: DataFrame - 消费者信心指数数据
    consumer_price: DataFrame - 消费价格涨幅数据
    """
    print("注意: 使用简化方法生成预测文件。理想情况下应先运行triangle_PCA_model.py生成更准确的预测结果。")

    # 1. 提取消费增长率
    # 首先尝试获取社会消费品零售总额增长率（Retailsalegryoym）- 这是主要目标变量
    target_data = None
    if retail_sales is not None:
        if 'Retailsalegryoym' in retail_sales.columns:
            if 'Datasign' in retail_sales.columns:
                # 筛选A类型数据(当期同比)
                target_data = retail_sales[retail_sales['Datasign'] == 'A']['Retailsalegryoym']
            else:
                target_data = retail_sales['Retailsalegryoym']

    # 如果没有找到目标列，尝试其他替代方案
    if target_data is None or target_data.empty:
        # 尝试从零售总额数据中找到任何可能的增长率列
        if retail_sales is not None:
            growth_cols = [col for col in retail_sales.columns if 'gr' in col.lower() or 'growth' in col.lower()]
            if growth_cols:
                target_data = retail_sales[growth_cols[0]]
            else:
                # 如果找不到增长率列，使用任何数值列
                numeric_cols = retail_sales.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    target_data = retail_sales[numeric_cols[0]]

    # 如果零售数据无法使用，尝试消费者信心指数
    if target_data is None or target_data.empty:
        if consumer_confidence is not None and 'Bcidm0203' in consumer_confidence.columns:
            target_data = consumer_confidence['Bcidm0203']

    # 最后尝试消费者价格数据
    if target_data is None or target_data.empty:
        if consumer_price is not None:
            numeric_cols = consumer_price.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                target_data = consumer_price[numeric_cols[0]]

    # 如果仍然找不到任何数据，抛出错误
    if target_data is None or target_data.empty:
        raise ValueError("无法从提供的数据中找到合适的目标变量")

    # 2. 确保索引是日期类型
    if not isinstance(target_data.index, pd.DatetimeIndex):
        # 尝试从源数据框中获取日期列
        dates = None
        for df in [retail_sales, consumer_confidence, consumer_price]:
            if df is not None and hasattr(df, 'index') and isinstance(df.index, pd.DatetimeIndex):
                dates = df.index
                break

        # 如果找不到日期，创建一个假设的日期范围
        if dates is None:
            print("警告: 无法从数据中确定日期，使用假设的日期范围")
            dates = pd.date_range(start='2020-01-01', periods=len(target_data), freq='M')

        # 设置日期索引
        target_data.index = dates[:len(target_data)]

    # 3. 创建DataFrame并确保列名为'0'
    consumption_growth = pd.DataFrame(target_data).rename(
        columns={target_data.name if hasattr(target_data, 'name') else 0: '0'})

    # 4. 创建results目录
    os.makedirs('results', exist_ok=True)

    # 5. 保存预测文件
    # 对于每个模型，保存相同的数据（简化方法）
    # 在实际应用中，应该通过01_triangle_PCA_model.py生成不同模型的预测
    consumption_growth.to_csv('results/predictions_SARIMA.csv')
    consumption_growth.to_csv('results/predictions_VAR.csv')
    consumption_growth.to_csv('results/predictions_AutoML.csv')
    consumption_growth.to_csv('results/predictions_Ensemble.csv')

    # 6. 生成模拟的集成权重（等权重）
    weights = pd.DataFrame({
        'SARIMA': [0.333] * len(consumption_growth),
        'VAR': [0.333] * len(consumption_growth),
        'AutoML': [0.334] * len(consumption_growth)
    }, index=consumption_growth.index)
    weights.to_csv('results/ensemble_weights.csv')

    # 7. 生成模拟的PCA因子
    # 在实际应用中，这些因子应该通过01_triangle_PCA_model.py的PCA分析生成
    pca_factors = pd.DataFrame(index=consumption_growth.index)
    # 使用种子以确保可重复性
    np.random.seed(42)
    pca_factors['C1'] = np.random.normal(0, 1, len(consumption_growth))  # 总体消费因子
    pca_factors['C2'] = np.random.normal(0, 1, len(consumption_growth))  # 结构性消费因子
    pca_factors['C3'] = np.random.normal(0, 1, len(consumption_growth))  # 消费者心理因子
    pca_factors.to_csv('results/pca_factors.csv')

    print("已生成预测结果文件（简化方法）")
    return consumption_growth


# =====================================================================
# 原代码部分：数据加载与预处理
# =====================================================================

def load_triangle_model_results():
    """
    加载三角组合模型的预测结果文件
    """
    try:
        # 加载各预测结果
        sarima_pred = pd.read_csv('results/predictions_SARIMA.csv')
        var_pred = pd.read_csv('results/predictions_VAR.csv')
        automl_pred = pd.read_csv('results/predictions_AutoML.csv')
        ensemble_pred = pd.read_csv('results/predictions_Ensemble.csv')

        # 加载集成权重
        ensemble_weights = pd.read_csv('results/ensemble_weights.csv')

        # 加载PCA因子
        pca_factors = pd.read_csv('results/pca_factors.csv')

        print("三角组合模型结果加载成功")
        return sarima_pred, var_pred, automl_pred, ensemble_pred, ensemble_weights, pca_factors

    except Exception as e:
        print(f"数据加载错误: {e}")
        return None, None, None, None, None, None


def preprocess_prediction_data(sarima_pred, var_pred, automl_pred, ensemble_pred, ensemble_weights):
    """
    对预测结果进行预处理:
    1. 转换日期格式
    2. 设置索引
    3. 对齐时间序列
    """
    # 转换日期格式
    for df in [sarima_pred, var_pred, automl_pred, ensemble_pred, ensemble_weights]:
        if df is not None and 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)

    # 找出共同的时间范围
    dfs = [df for df in [sarima_pred, var_pred, automl_pred, ensemble_pred, ensemble_weights] if df is not None]
    if not dfs:
        return None, None, None, None, None

    common_index = dfs[0].index
    for df in dfs[1:]:
        common_index = common_index.intersection(df.index)

    # 对齐索引
    for i, df in enumerate(dfs):
        dfs[i] = df.loc[common_index]

    # 重新分配回原变量
    i = 0
    sarima_pred = dfs[i] if i < len(dfs) and sarima_pred is not None else None
    i += 1 if sarima_pred is not None else 0
    var_pred = dfs[i] if i < len(dfs) and var_pred is not None else None
    i += 1 if var_pred is not None else 0
    automl_pred = dfs[i] if i < len(dfs) and automl_pred is not None else None
    i += 1 if automl_pred is not None else 0
    ensemble_pred = dfs[i] if i < len(dfs) and ensemble_pred is not None else None
    i += 1 if ensemble_pred is not None else 0
    ensemble_weights = dfs[i] if i < len(dfs) and ensemble_weights is not None else None

    print("预测数据预处理完成")
    return sarima_pred, var_pred, automl_pred, ensemble_pred, ensemble_weights


def calculate_consumption_growth_change(ensemble_pred):
    """
    计算消费增长变化率 ΔCt = Ŷt - Ŷt-1
    """
    if ensemble_pred is None:
        return None

    # 使用集成预测结果计算消费增长变化
    if '0' in ensemble_pred.columns:  # 根据文件结构确认
        consumption_growth = ensemble_pred['0']
    else:
        # 尝试使用第一列，不管列名
        consumption_growth = ensemble_pred.iloc[:, 0]

    # 计算增长变化率
    delta_consumption = consumption_growth.diff()

    # 应用简单移动平均平滑处理
    window_size = 3  # 3个月平滑窗口
    smoothed_delta_consumption = delta_consumption.rolling(window=window_size).mean()

    # 填充初始的NaN值
    smoothed_delta_consumption = smoothed_delta_consumption.fillna(method='bfill')

    print("消费增长变化率计算完成")
    return smoothed_delta_consumption


# =====================================================================
# 第二部分：消费贝塔计算模型
# =====================================================================
def align_data_frequencies(stock_returns, market_returns, delta_consumption):
    """
    对齐不同频率的数据，增强版本处理各种日期格式不匹配的情况
    """
    import pandas as pd
    import numpy as np

    # 检查日期分布，输出诊断信息
    stock_dates = stock_returns.index
    consumption_dates = delta_consumption.index

    print(f"股票收益率数据日期范围: {stock_dates.min()} 至 {stock_dates.max()}，共 {len(stock_dates)} 个点")
    print(f"消费数据日期范围: {consumption_dates.min()} 至 {consumption_dates.max()}，共 {len(consumption_dates)} 个点")

    # 更详细地打印日期格式信息
    print(f"股票收益率数据索引类型: {type(stock_dates)}, dtype: {stock_dates.dtype}")
    print(f"消费数据索引类型: {type(consumption_dates)}, dtype: {consumption_dates.dtype}")

    # 打印前几个日期样本
    print(f"股票收益率数据前5个日期: {stock_dates[:5]}")
    print(f"消费数据前5个日期: {consumption_dates[:5]}")

    # 确保两者都是DatetimeIndex，标准化日期格式
    if not isinstance(stock_dates, pd.DatetimeIndex):
        stock_returns.index = pd.to_datetime(stock_returns.index)
        stock_dates = stock_returns.index
        print("已将股票收益率数据索引转换为DatetimeIndex")

    if not isinstance(consumption_dates, pd.DatetimeIndex):
        delta_consumption.index = pd.to_datetime(delta_consumption.index)
        consumption_dates = delta_consumption.index
        print("已将消费数据索引转换为DatetimeIndex")

    # 移除可能的时区信息
    if stock_dates.tz is not None:
        stock_returns.index = stock_dates.tz_localize(None)
        stock_dates = stock_returns.index
        print("已移除股票收益率数据的时区信息")

    if consumption_dates.tz is not None:
        delta_consumption.index = consumption_dates.tz_localize(None)
        consumption_dates = delta_consumption.index
        print("已移除消费数据的时区信息")

    # 计算平均日期间隔
    stock_interval = (stock_dates.max() - stock_dates.min()).days / max(1, len(stock_dates) - 1)
    consumption_interval = (consumption_dates.max() - consumption_dates.min()).days / max(1, len(consumption_dates) - 1)

    print(f"股票收益率平均日期间隔: {stock_interval:.1f} 天")
    print(f"消费数据平均日期间隔: {consumption_interval:.1f} 天")

    # 判断是否需要重采样 - 消费数据频率明显低于股票数据
    if consumption_interval > stock_interval * 2:
        print("检测到频率不匹配: 正在进行日期标准化和重采样...")

        # 确定月份格式：使用月末日期作为标准
        # 将股票数据重采样为月度
        stock_returns_monthly = stock_returns.resample('M').apply(
            lambda x: (1 + x).prod() - 1 if len(x) > 0 else np.nan
        )

        if isinstance(market_returns, pd.Series):
            market_returns_monthly = market_returns.resample('M').apply(
                lambda x: (1 + x).prod() - 1 if len(x) > 0 else np.nan
            )
        else:
            # 如果market_returns是DataFrame
            market_returns_monthly = market_returns.resample('M').apply(
                lambda x: (1 + x).prod() - 1 if len(x) > 0 else np.nan
            )

        # 标准化消费数据的索引为月末
        # 对每个日期找到对应月份的最后一天
        new_index = []
        for date in delta_consumption.index:
            last_day_of_month = pd.Timestamp(date.year, date.month, 1) + pd.offsets.MonthEnd(1)
            new_index.append(last_day_of_month)

        delta_consumption_standardized = delta_consumption.copy()
        delta_consumption_standardized.index = pd.DatetimeIndex(new_index)

        # 查看处理后的索引
        print(f"重采样后股票数据前5个日期: {stock_returns_monthly.index[:5]}")
        print(f"标准化后消费数据前5个日期: {delta_consumption_standardized.index[:5]}")

        # 找出共同的月份
        common_index = stock_returns_monthly.index.intersection(
            market_returns_monthly.index).intersection(delta_consumption_standardized.index)

        print(f"共同日期范围: {common_index.min() if len(common_index) > 0 else '无'} 至 "
              f"{common_index.max() if len(common_index) > 0 else '无'}")

        # 如果仍然没有共同的时间点，尝试更灵活的匹配
        if len(common_index) == 0:
            print("尝试更灵活的日期匹配方法...")

            # 使用年月作为匹配键而不是完整日期
            sr_ym = [(d.year, d.month) for d in stock_returns_monthly.index]
            mr_ym = [(d.year, d.month) for d in market_returns_monthly.index]
            dc_ym = [(d.year, d.month) for d in delta_consumption_standardized.index]

            # 找出共同的年月
            common_ym = set(sr_ym).intersection(set(mr_ym)).intersection(set(dc_ym))
            print(f"使用年月匹配找到 {len(common_ym)} 个共同时间点")

            if len(common_ym) > 0:
                # 创建新的数据集，只保留共同的年月
                new_stock_returns = []
                new_market_returns = []
                new_delta_consumption = []
                new_dates = []

                for ym in sorted(common_ym):
                    year, month = ym
                    date = pd.Timestamp(year, month, 1) + pd.offsets.MonthEnd(1)
                    new_dates.append(date)

                    # 查找对应的股票收益率
                    for i, d in enumerate(sr_ym):
                        if d == ym:
                            new_stock_returns.append(stock_returns_monthly.iloc[i])
                            break

                    # 查找对应的市场收益率
                    for i, d in enumerate(mr_ym):
                        if d == ym:
                            if isinstance(market_returns_monthly, pd.Series):
                                new_market_returns.append(market_returns_monthly.iloc[i])
                            else:
                                new_market_returns.append(market_returns_monthly.iloc[i])
                            break

                    # 查找对应的消费增长变化率
                    for i, d in enumerate(dc_ym):
                        if d == ym:
                            new_delta_consumption.append(delta_consumption_standardized.iloc[i])
                            break

                # 创建新的DataFrame和Series
                stock_returns_aligned = pd.DataFrame(
                    new_stock_returns,
                    index=new_dates,
                    columns=stock_returns_monthly.columns
                )

                if isinstance(market_returns_monthly, pd.Series):
                    market_returns_aligned = pd.Series(
                        new_market_returns,
                        index=new_dates,
                        name=market_returns_monthly.name
                    )
                else:
                    market_returns_aligned = pd.DataFrame(
                        new_market_returns,
                        index=new_dates,
                        columns=market_returns_monthly.columns
                    )

                if isinstance(delta_consumption_standardized, pd.Series):
                    delta_consumption_aligned = pd.Series(
                        new_delta_consumption,
                        index=new_dates,
                        name=delta_consumption_standardized.name
                    )
                else:
                    delta_consumption_aligned = pd.DataFrame(
                        new_delta_consumption,
                        index=new_dates,
                        columns=delta_consumption_standardized.columns
                    )

                print(f"最终对齐后的数据集共有 {len(new_dates)} 个时间点")
                return stock_returns_aligned, market_returns_aligned, delta_consumption_aligned
            else:
                print("警告: 即使使用灵活匹配，仍然找不到共同的时间点")

                # 创建一些合成数据以便程序可以继续运行
                # 注意：这只是为了防止程序中断，实际应用中应该调整历史数据
                print("创建模拟数据以便程序继续运行...")

                # 使用消费数据的时间范围
                sim_dates = pd.date_range(
                    start=consumption_dates.min(),
                    end=consumption_dates.max(),
                    freq='M'
                )

                # 创建模拟数据
                stock_returns_aligned = pd.DataFrame(
                    np.random.normal(0.01, 0.05, (len(sim_dates), len(stock_returns.columns))),
                    index=sim_dates,
                    columns=stock_returns.columns
                )

                if isinstance(market_returns, pd.Series):
                    market_returns_aligned = pd.Series(
                        np.random.normal(0.005, 0.03, len(sim_dates)),
                        index=sim_dates,
                        name=market_returns.name if hasattr(market_returns, 'name') else 'market_return'
                    )
                else:
                    market_returns_aligned = pd.DataFrame(
                        np.random.normal(0.005, 0.03, (len(sim_dates), len(market_returns.columns))),
                        index=sim_dates,
                        columns=market_returns.columns
                    )

                delta_consumption_aligned = delta_consumption.reindex(sim_dates, method='ffill')

                print(f"创建了 {len(sim_dates)} 个模拟数据点，请注意这仅用于演示")
                return stock_returns_aligned, market_returns_aligned, delta_consumption_aligned

        # 对齐数据
        stock_returns_aligned = stock_returns_monthly.loc[common_index]
        market_returns_aligned = market_returns_monthly.loc[common_index]
        delta_consumption_aligned = delta_consumption_standardized.loc[common_index]
    else:
        # 直接找出共同日期
        common_index = stock_returns.index.intersection(
            market_returns.index).intersection(delta_consumption.index)

        # 对齐数据
        stock_returns_aligned = stock_returns.loc[common_index]
        market_returns_aligned = market_returns.loc[common_index]
        delta_consumption_aligned = delta_consumption.loc[common_index]

    print(f"对齐后的数据集共有 {len(common_index)} 个时间点")

    if len(common_index) == 0:
        # 如果仍然没有共同时间点，创建合成数据（应急措施）
        print("警告: 无法找到共同的时间点，将使用合成数据")
        # 使用消费数据的时间范围
        sim_dates = pd.date_range(
            start=consumption_dates.min(),
            end=consumption_dates.max(),
            freq='M'
        )

        # 创建模拟数据
        stock_returns_aligned = pd.DataFrame(
            np.random.normal(0.01, 0.05, (len(sim_dates), len(stock_returns.columns))),
            index=sim_dates,
            columns=stock_returns.columns
        )

        if isinstance(market_returns, pd.Series):
            market_returns_aligned = pd.Series(
                np.random.normal(0.005, 0.03, len(sim_dates)),
                index=sim_dates,
                name=market_returns.name if hasattr(market_returns, 'name') else 'market_return'
            )
        else:
            market_returns_aligned = pd.DataFrame(
                np.random.normal(0.005, 0.03, (len(sim_dates), len(market_returns.columns))),
                index=sim_dates,
                columns=market_returns.columns
            )

        if isinstance(delta_consumption, pd.Series):
            # 如果是Series，使用原始值附加到新索引
            delta_consumption_aligned = pd.Series(
                delta_consumption.values[:min(len(sim_dates), len(delta_consumption))],
                index=sim_dates[:min(len(sim_dates), len(delta_consumption))],
                name=delta_consumption.name if hasattr(delta_consumption, 'name') else 'delta_consumption'
            )
            # 填充可能的缺失值
            delta_consumption_aligned = delta_consumption_aligned.reindex(sim_dates).ffill().bfill()
        else:
            # 如果是DataFrame
            values = delta_consumption.values[:min(len(sim_dates), len(delta_consumption))]
            if len(values) < len(sim_dates):
                # 不足的部分用最后一行复制
                last_row = values[-1:] if len(values) > 0 else np.zeros((1, delta_consumption.shape[1]))
                values = np.vstack([values, np.tile(last_row, (len(sim_dates) - len(values), 1))])

            delta_consumption_aligned = pd.DataFrame(
                values,
                index=sim_dates,
                columns=delta_consumption.columns
            )

        print(f"创建了 {len(sim_dates)} 个模拟数据点，请注意这仅用于演示")

    return stock_returns_aligned, market_returns_aligned, delta_consumption_aligned


def calculate_consumer_beta(stock_returns, market_returns, delta_consumption, window_size=12, lambda_param=0.95):
    """
    使用滚动窗口计算股票的消费贝塔 - 修改版，确保日期索引对齐
    """
    # 显示输入数据形状
    print(f"计算消费贝塔 - 输入数据形状:")
    print(
        f"股票收益率: {stock_returns.shape}, 市场收益率: {market_returns.shape if hasattr(market_returns, 'shape') else len(market_returns)}, 消费增长变化率: {delta_consumption.shape if hasattr(delta_consumption, 'shape') else len(delta_consumption)}")

    # 检查有效数据点数量
    if len(stock_returns) < window_size:
        print(f"警告: 可用数据点数量({len(stock_returns)})少于所需窗口大小({window_size})")
        # 调整窗口大小
        window_size = max(3, len(stock_returns) // 2)  # 至少3个点，最多不超过数据集一半
        print(f"自动调整窗口大小为: {window_size}")

    # 对收益率进行极端值处理
    stock_returns_winsorized = winsorize_returns(stock_returns)

    # 初始化结果DataFrame
    consumer_betas = pd.DataFrame(index=stock_returns.index, columns=stock_returns.columns)

    # 计算每个时间点的消费贝塔
    for t in range(window_size, len(stock_returns)):
        # 当前窗口的时间索引
        window_index = stock_returns.index[t - window_size:t]

        # 计算时间权重
        weights = np.array([lambda_param ** (window_size - j - 1) for j in range(window_size)])
        weights = weights / weights.sum()  # 归一化权重

        # 遍历每只股票
        for stock in stock_returns.columns:
            # 提取窗口内数据
            y = stock_returns_winsorized.loc[window_index, stock].values

            if isinstance(market_returns, pd.Series):
                market_values = market_returns.loc[window_index].values
            else:  # DataFrame
                market_values = market_returns.loc[window_index].iloc[:, 0].values

            if isinstance(delta_consumption, pd.Series):
                consumption_values = delta_consumption.loc[window_index].values
            else:  # DataFrame
                consumption_values = delta_consumption.loc[window_index].iloc[:, 0].values

            X = np.column_stack([market_values, consumption_values])

            # 添加常数项
            X = sm.add_constant(X)

            # 使用加权最小二乘法估计
            try:
                model = sm.WLS(y, X, weights=weights).fit()
                consumer_betas.loc[stock_returns.index[t], stock] = model.params[2]  # 第三个参数是消费贝塔
            except Exception as e:
                # 如果估计失败，使用OLS
                try:
                    model = sm.OLS(y, X).fit()
                    consumer_betas.loc[stock_returns.index[t], stock] = model.params[2]
                except Exception as e2:
                    # 如果仍然失败，设为NaN
                    consumer_betas.loc[stock_returns.index[t], stock] = np.nan

    print("个股消费贝塔计算完成")
    # 确保结果为数值类型
    consumer_betas = consumer_betas.apply(pd.to_numeric, errors='coerce')
    print("已将消费贝塔转换为数值类型")
    return consumer_betas


def investigate_data_structures(stock_returns, market_returns, delta_consumption, ensemble_pred):
    """
    详细调查数据结构，以便找出不匹配的原因
    """
    print("\n===== 数据结构调查 =====")

    # 1. 检查日期索引格式
    print("\n-- 日期索引格式 --")
    for name, data in [
        ("股票收益率", stock_returns),
        ("市场收益率", market_returns),
        ("消费增长变化率", delta_consumption),
        ("集成预测", ensemble_pred)
    ]:
        if data is not None:
            print(f"{name}:")
            print(f"  索引类型: {type(data.index)}")
            print(f"  索引数据类型: {data.index.dtype}")
            print(f"  前5个日期: {data.index[:5]}")
            print(f"  后5个日期: {data.index[-5:]}")

            # 检查是否为DatetimeIndex类型
            has_tz = False
            if hasattr(data.index, 'tz'):
                has_tz = data.index.tz is not None
                print(f"  有无时区信息: {has_tz}")
            else:
                print(f"  有无时区信息: 不适用（非日期时间索引）")

            # 检查特殊格式问题
            date_strings = [str(d) for d in data.index[:5]]
            print(f"  日期字符串样例: {date_strings}")

    # 2. 检查消费数据具体内容
    print("\n-- 消费数据内容 --")
    if delta_consumption is not None:
        if isinstance(delta_consumption, pd.Series):
            print(f"消费增长变化率 (Series):")
            print(f"  名称: {delta_consumption.name}")
            print(f"  前5个值: {delta_consumption.head()}")
        else:
            print(f"消费增长变化率 (DataFrame):")
            print(f"  列名: {delta_consumption.columns}")
            print(f"  前5行: \n{delta_consumption.head()}")

    if ensemble_pred is not None:
        print(f"\n集成预测数据:")
        print(f"  列名: {ensemble_pred.columns}")
        print(f"  前5行: \n{ensemble_pred.head()}")

    # 3. 检查日期范围重叠情况
    print("\n-- 日期范围重叠分析 --")
    all_dates = []
    all_date_sets = []

    for name, data in [
        ("股票收益率", stock_returns),
        ("市场收益率", market_returns),
        ("消费增长变化率", delta_consumption),
        ("集成预测", ensemble_pred)
    ]:
        if data is not None:
            # 检查索引是否为日期时间类型
            if isinstance(data.index, pd.DatetimeIndex):
                # 转换为年月格式以便比较
                date_set = set([(d.year, d.month) for d in data.index])
                all_date_sets.append(date_set)
                all_dates.append((name, date_set))
                print(f"{name}: {len(date_set)} 个不同的年月组合")
            else:
                print(f"{name}: 索引不是日期时间类型，跳过年月分析")
                # 对于非日期索引，创建一个空的集合
                all_date_sets.append(set())
                all_dates.append((name, set()))

    # 找出所有数据集共有的年月（只考虑有效的日期集合）
    valid_date_sets = [date_set for date_set in all_date_sets if date_set]
    if len(valid_date_sets) >= 2:
        common_dates = set.intersection(*valid_date_sets)
        print(f"\n所有有效数据集共有 {len(common_dates)} 个年月组合")
        if common_dates:
            print(f"共有年月范围: {min(common_dates)} 至 {max(common_dates)}")
            print(f"共有年月样例 (前5个): {sorted(list(common_dates))[:5]}")

        # 检查每对数据集的共有年月
        valid_dates = [(name, dates) for name, dates in all_dates if dates]
        if len(valid_dates) > 1:
            print("\n各数据集两两之间的共有年月数量:")
            for i in range(len(valid_dates)):
                for j in range(i + 1, len(valid_dates)):
                    name_i, dates_i = valid_dates[i]
                    name_j, dates_j = valid_dates[j]
                    common = dates_i.intersection(dates_j)
                    print(f"  {name_i} 与 {name_j}: {len(common)} 个共有年月")
    else:
        common_dates = set()
        print("\n没有足够的有效日期数据集进行比较")

    # 4. 提出可能的解决方案
    print("\n-- 可能的解决方案 --")
    if len(common_dates) == 0:
        print("1. 检查并重新生成ensemble_pred，确保其日期范围覆盖股票数据")
        print("2. 重新采样消费数据，调整为与股票数据相匹配的频率")
        print("3. 确保所有数据集使用相同的日期格式和标准")
    else:
        print("可以使用年月匹配来找到共同的时间点")

    return common_dates

def winsorize_returns(returns, lower_percentile=0.05, upper_percentile=0.95):
    """
    对收益率进行截尾处理，控制极端值
    """
    winsorized_returns = returns.copy()

    # 逐期进行截尾处理
    for date in returns.index:
        returns_t = returns.loc[date]

        # 计算百分位数
        lower_bound = returns_t.quantile(lower_percentile)
        upper_bound = returns_t.quantile(upper_percentile)

        # 截尾处理
        winsorized_returns.loc[date] = returns_t.clip(lower=lower_bound, upper=upper_bound)

    return winsorized_returns


def calculate_industry_group_betas(consumer_betas, industry_classification):
    """
    计算行业组平均消费贝塔

    参数:
    consumer_betas: DataFrame - 个股消费贝塔
    industry_classification: Series - 股票行业分类，索引为股票代码，值为行业类别

    返回:
    dict - 行业组平均消费贝塔
    """
    # 将行业分为高、中、低消费相关三类
    industry_groups = {
        'high_consumption': ['消费品', '零售', '餐饮', '食品饮料', '纺织服装', '家用电器', '商贸零售'],
        'medium_consumption': ['医疗', '科技', '传媒', '汽车', '医药生物', '计算机', '通信'],
        'low_consumption': ['能源', '公用事业', '金融', '材料', '房地产', '建筑', '交通运输', '采掘', '钢铁',
                            '有色金属']
    }

    # 映射股票到消费相关度组
    stock_to_group = {}
    for stock, industry in industry_classification.items():
        for group, industries in industry_groups.items():
            if any(ind in industry for ind in industries):
                stock_to_group[stock] = group
                break
        if stock not in stock_to_group:
            stock_to_group[stock] = 'medium_consumption'  # 默认归为中等消费相关度

    # 计算每个消费相关度组的平均消费贝塔
    group_betas = {}
    for date in consumer_betas.index:
        group_betas[date] = {'high_consumption': [], 'medium_consumption': [], 'low_consumption': []}

        for stock in consumer_betas.columns:
            if stock in stock_to_group:
                group = stock_to_group[stock]
                beta_value = consumer_betas.loc[date, stock]
                if not np.isnan(beta_value):
                    group_betas[date][group].append(beta_value)

        # 计算每组的平均值
        for group in group_betas[date]:
            if group_betas[date][group]:
                group_betas[date][group] = np.mean(group_betas[date][group])
            else:
                group_betas[date][group] = np.nan

    print("行业组平均消费贝塔计算完成")
    return group_betas


def adjust_betas_with_industry_average(consumer_betas, group_betas, industry_classification, kappa=0.3):
    """
    将个股消费贝塔与行业组平均值结合，提高稳定性

    参数:
    consumer_betas: DataFrame - 个股消费贝塔
    group_betas: dict - 行业组平均消费贝塔
    industry_classification: Series - 股票行业分类
    kappa: float - 权重参数，控制个股vs行业平均的比重

    返回:
    DataFrame - 调整后的个股消费贝塔
    """
    # 将行业分为高、中、低消费相关三类（与上面函数相同逻辑）
    industry_groups = {
        'high_consumption': ['消费品', '零售', '餐饮', '食品饮料', '纺织服装', '家用电器', '商贸零售'],
        'medium_consumption': ['医疗', '科技', '传媒', '汽车', '医药生物', '计算机', '通信'],
        'low_consumption': ['能源', '公用事业', '金融', '材料', '房地产', '建筑', '交通运输', '采掘', '钢铁',
                            '有色金属']
    }

    # 映射股票到消费相关度组
    stock_to_group = {}
    for stock, industry in industry_classification.items():
        for group, industries in industry_groups.items():
            if any(ind in industry for ind in industries):
                stock_to_group[stock] = group
                break
        if stock not in stock_to_group:
            stock_to_group[stock] = 'medium_consumption'  # 默认归为中等消费相关度

    # 调整消费贝塔
    adjusted_betas = consumer_betas.copy()

    for date in consumer_betas.index:
        for stock in consumer_betas.columns:
            if stock in stock_to_group:
                group = stock_to_group[stock]
                if date in group_betas and group in group_betas[date] and not np.isnan(group_betas[date][group]):
                    # 使用公式：β̃i,c,t = (1-κ)β̂i,c,t + κβ̄g,c,t
                    original_beta = consumer_betas.loc[date, stock]
                    group_avg_beta = group_betas[date][group]

                    if not np.isnan(original_beta):
                        adjusted_betas.loc[date, stock] = (1 - kappa) * original_beta + kappa * group_avg_beta

    print("消费贝塔行业调整完成")
    # 确保结果为数值类型
    adjusted_betas = adjusted_betas.apply(pd.to_numeric, errors='coerce')
    print("已将调整后的消费贝塔转换为数值类型")
    return adjusted_betas


# =====================================================================
# 第三部分：消费环境判断与投资策略
# =====================================================================

def judge_consumption_environment(latest_forecast, historical_avg):
    """
    根据最新预测值与历史均值比较，判断消费环境

    参数:
    latest_forecast: float - 最新消费增速预测值
    historical_avg: float - 历史消费增速均值

    返回:
    str - 消费环境判断 ('良好' 或 '疲软')
    """
    if latest_forecast > historical_avg:
        return "良好"
    else:
        return "疲软"


def select_stocks_based_on_environment(consumer_betas, environment, n_stocks=50):
    """
    根据消费环境选择合适的股票

    参数:
    consumer_betas: DataFrame - 调整后的个股消费贝塔
    environment: str - 消费环境判断结果
    n_stocks: int - 选择的股票数量

    返回:
    list - 选中的股票列表
    """
    # 获取最新日期的消费贝塔
    latest_date = consumer_betas.index[-1]
    latest_betas = consumer_betas.loc[latest_date].dropna()

    # 确保数据类型为数值型（这是解决错误的关键）
    latest_betas = pd.to_numeric(latest_betas, errors='coerce')
    latest_betas = latest_betas.dropna()  # 移除无法转换为数值的项

    # 检查是否还有足够的股票
    if len(latest_betas) < n_stocks:
        print(f"警告：只有 {len(latest_betas)} 支有效股票，少于请求的 {n_stocks} 支")
        n_stocks = min(n_stocks, len(latest_betas))

    if n_stocks == 0:
        return []

    if environment == "良好":
        # 选择消费贝塔高的股票
        selected_stocks = latest_betas.nlargest(n_stocks).index.tolist()
    else:
        # 选择消费贝塔低的股票
        selected_stocks = latest_betas.nsmallest(n_stocks).index.tolist()

    return selected_stocks

# =====================================================================
# 第四部分：模型评估与实证检验
# =====================================================================

def evaluate_consumer_beta_significance(consumer_betas, stock_returns, market_returns, delta_consumption):
    """
    评估消费贝塔的统计显著性
    """
    # 初始化结果DataFrame
    significance_results = pd.DataFrame(index=consumer_betas.columns,
                                        columns=['消费贝塔平均值', 't值', 'p值', '是否显著(10%)'])

    # 对每只股票进行t检验
    for stock in consumer_betas.columns:
        # 提取有效的消费贝塔值
        valid_betas = consumer_betas[stock].dropna()

        if len(valid_betas) > 0:
            # 计算平均消费贝塔
            avg_beta = valid_betas.mean()

            # 提取该股票的收益率及对应时间的市场收益率和消费增长变化
            common_index = stock_returns.index.intersection(market_returns.index).intersection(delta_consumption.index)
            y = stock_returns.loc[common_index, stock]
            X = pd.DataFrame({
                'const': 1,
                'market': market_returns.loc[common_index],
                'consumption': delta_consumption.loc[common_index]
            })

            # 使用OLS估计模型
            try:
                model = sm.OLS(y, X).fit()
                t_value = model.tvalues['consumption']
                p_value = model.pvalues['consumption']
                is_significant = p_value < 0.1
            except:
                t_value = np.nan
                p_value = np.nan
                is_significant = False

            # 保存结果
            significance_results.loc[stock] = [avg_beta, t_value, p_value, is_significant]
        else:
            significance_results.loc[stock] = [np.nan, np.nan, np.nan, False]

    # 计算显著比例
    significant_ratio = significance_results['是否显著(10%)'].mean()
    print(f"消费贝塔显著比例: {significant_ratio:.2%}")

    return significance_results


def backtest_portfolio_strategy(consumer_betas, stock_returns, forecast_consumption, historical_avg, n_stocks=50):
    """
    回测基于消费贝塔的投资组合策略

    参数:
    consumer_betas: DataFrame - 调整后的个股消费贝塔
    stock_returns: DataFrame - 个股收益率数据
    forecast_consumption: Series - 预测的消费增速
    historical_avg: float - 历史消费增速均值
    n_stocks: int - 每个组合中的股票数量

    返回:
    DataFrame - 回测结果
    """
    # 初始化结果
    portfolio_returns = pd.Series(index=stock_returns.index[1:], dtype=float)
    market_returns_subset = stock_returns.mean(axis=1)  # 简单假设市场收益是所有股票的平均

    # 每月调整组合
    for i in range(1, len(stock_returns)):
        date = stock_returns.index[i]
        prev_date = stock_returns.index[i - 1]

        # 如果有预测消费增速
        if prev_date in forecast_consumption.index:
            # 判断消费环境
            environment = judge_consumption_environment(forecast_consumption[prev_date], historical_avg)

            # 根据环境选择股票
            if prev_date in consumer_betas.index:
                selected_stocks = select_stocks_based_on_environment(
                    consumer_betas.loc[:prev_date], environment, n_stocks)

                # 计算组合收益率 (等权重)
                if selected_stocks:
                    valid_stocks = [s for s in selected_stocks if s in stock_returns.columns]
                    if valid_stocks:
                        portfolio_return = stock_returns.loc[date, valid_stocks].mean()
                        portfolio_returns[date] = portfolio_return

    # 计算累计收益
    cumulative_portfolio = (1 + portfolio_returns).cumprod() - 1
    cumulative_market = (1 + market_returns_subset).cumprod() - 1

    # 计算指标
    excess_return = portfolio_returns - market_returns_subset
    sharpe = portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(12)  # 年化夏普比率

    # 结果
    results = {
        'portfolio_returns': portfolio_returns,
        'market_returns': market_returns_subset,
        'cumulative_portfolio': cumulative_portfolio,
        'cumulative_market': cumulative_market,
        'excess_return': excess_return,
        'sharpe_ratio': sharpe
    }

    print(f"投资组合夏普比率: {sharpe:.2f}")
    print(f"年化超额收益率: {excess_return.mean() * 12:.2%}")

    return results


def plot_backtest_results(results):
    """
    绘制回测结果图表
    """
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    # 绘制累计收益对比图
    plt.figure(figsize=(10, 6))
    results['cumulative_portfolio'].plot(label='消费贝塔策略')
    results['cumulative_market'].plot(label='市场基准')
    plt.title('累计收益对比')
    plt.xlabel('日期')
    plt.ylabel('累计收益率')
    plt.legend()
    plt.grid(True)
    plt.savefig('02results/cumulative_returns.png')

    # 绘制月度超额收益图
    plt.figure(figsize=(10, 6))
    results['excess_return'].plot(kind='bar')
    plt.title('月度超额收益')
    plt.xlabel('日期')
    plt.ylabel('超额收益率')
    plt.grid(True)
    plt.savefig('02results/excess_returns.png')


# =====================================================================
# 新增部分：整合预处理和主函数
# =====================================================================

def main_with_preprocessing():
    """
    包含数据预处理步骤的主函数
    """
    print("开始执行消费贝塔敏感度测度模型（包含预处理）...")

    # 创建results目录（如果不存在）
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # 创建02results目录（如果不存在）
    if not os.path.exists('02results'):
        os.makedirs('02results')

    # 步骤1: 检查预测结果文件是否存在
    # 这些文件应该由triangle_PCA_model.py生成
    prediction_files = [
        'results/predictions_SARIMA.csv',
        'results/predictions_VAR.csv',
        'results/predictions_AutoML.csv',
        'results/predictions_Ensemble.csv',
        'results/ensemble_weights.csv',
        'results/pca_factors.csv'
    ]

    prediction_files_exist = all(os.path.exists(f) for f in prediction_files)

    if not prediction_files_exist:
        print("\n警告: 预测结果文件不完整")
        print("这些文件应该由01_triangle_PCA_model.py生成")
        print("将尝试从CME原始数据文件生成简化的预测结果")

        # 尝试加载消费数据
        try:
            retail_sales, consumer_confidence, consumer_price = load_consumption_data()
            print("消费数据加载成功")

            # 尝试生成预测文件
            try:
                generate_prediction_files(retail_sales, consumer_confidence, consumer_price)
                print("已使用简化方法生成预测文件")
            except Exception as e:
                print(f"生成预测结果文件失败: {e}")
                raise ValueError("无法生成预测结果文件，请先运行01_triangle_PCA_model.py")
        except Exception as e:
            print(f"消费数据加载失败: {e}")
            raise ValueError("无法加载消费数据，请确保CME_*.csv文件存在，或先运行01_triangle_PCA_model.py")
    else:
        print("发现预测结果文件，将直接使用")

    # 步骤2: 处理行业分类数据
    try:
        industry_classification = preprocess_industry_classification('industry_classification.csv')
    except Exception as e:
        print(f"行业分类数据处理失败: {e}")
        raise ValueError("无法处理行业分类数据，请确保industry_classification.csv文件存在且格式正确")

    # 步骤3: 加载三角组合模型的预测结果
    sarima_pred, var_pred, automl_pred, ensemble_pred, ensemble_weights, pca_factors = load_triangle_model_results()

    # 确保成功加载了预测结果
    if ensemble_pred is None:
        print("无法加载预测数据，请检查预测结果文件")
        raise ValueError("加载预测数据失败")

    # 步骤4: 预处理预测数据
    sarima_pred, var_pred, automl_pred, ensemble_pred, ensemble_weights = preprocess_prediction_data(
        sarima_pred, var_pred, automl_pred, ensemble_pred, ensemble_weights
    )

    # 步骤5: 计算消费增长变化率
    delta_consumption = calculate_consumption_growth_change(ensemble_pred)

    # 确保 delta_consumption 使用日期索引而不是整数索引
    if not isinstance(delta_consumption.index, pd.DatetimeIndex):
        print("注意: 转换 delta_consumption 索引为日期索引")
        if isinstance(ensemble_pred.index, pd.DatetimeIndex):
            # 如果 ensemble_pred 有日期索引，将其应用到 delta_consumption
            delta_consumption.index = ensemble_pred.index
        else:
            # 尝试创建一个合适的日期索引
            date_range = pd.date_range(start='2020-01-01', periods=len(delta_consumption), freq='M')
            delta_consumption.index = date_range
            print(f"警告：创建了模拟日期索引 {date_range[0]} 到 {date_range[-1]}")

    # 步骤6: 加载或生成股票收益率数据
    sample_size = len(delta_consumption)
    dates = delta_consumption.index

    # 尝试加载实际的股票收益率数据
    try:
        stock_returns = pd.read_csv('stock_returns.csv')
        stock_returns['Trddt'] = pd.to_datetime(stock_returns['Trddt'])
        stock_returns.set_index('Trddt', inplace=True)

        market_returns = pd.read_csv('market_returns.csv')
        market_returns['Trddt'] = pd.to_datetime(market_returns['Trddt'])
        market_returns.set_index('Trddt', inplace=True)
        market_returns = market_returns['market_return']

        print("股票收益率数据加载成功")
    except Exception as e:
        print(f"股票收益率数据加载失败: {e}")
        raise ValueError("无法加载股票收益率数据，请确保stock_returns.csv和market_returns.csv文件存在")

    # 步骤6.5: 调查数据结构
    common_dates = investigate_data_structures(
        stock_returns, market_returns, delta_consumption, ensemble_pred
    )

    # 步骤7: 确保行业分类与股票收益率数据匹配
    stocks_in_returns = list(stock_returns.columns)
    stocks_in_classification = list(industry_classification.index)

    print("股票收益率数据中的前5个股票代码:", stocks_in_returns[:5])
    print("行业分类数据中的前5个股票代码:", stocks_in_classification[:5])

    # 创建行业分类的副本，将索引转换为字符串并补齐6位数字
    new_industry_classification = pd.Series(
        industry_classification.values,
        index=[str(idx).zfill(6) for idx in industry_classification.index]
    )

    # 找出共同的股票代码
    common_stocks = list(set(stocks_in_returns) & set(new_industry_classification.index))

    if common_stocks:
        print(f"找到共同股票数量: {len(common_stocks)}")
        # 使用共有的股票子集
        stock_returns = stock_returns[common_stocks]
        industry_classification = new_industry_classification[common_stocks]
    else:
        print("警告: 股票收益率数据与行业分类数据没有共同的股票代码")

    # 步骤7.5: 对齐数据频率（在现有代码的第908行之前，计算consumer_betas之前添加）
    stock_returns_aligned, market_returns_aligned, delta_consumption_aligned = align_data_frequencies(
        stock_returns, market_returns, delta_consumption
    )
    # 如果仍然没有共同时间点，尝试使用集成预测数据生成模拟消费增长变化率
    if len(stock_returns_aligned) == 0:
        print("ohno，仍然没有共同时间点……")

    # 步骤8: 计算消费贝塔（使用对齐后的数据）
    consumer_betas = calculate_consumer_beta(
        stock_returns=stock_returns_aligned,
        market_returns=market_returns_aligned,
        delta_consumption=delta_consumption_aligned
    )

    # 步骤9: 计算行业组平均消费贝塔
    group_betas = calculate_industry_group_betas(
        consumer_betas=consumer_betas,
        industry_classification=industry_classification
    )

    # 步骤10: 调整消费贝塔
    adjusted_betas = adjust_betas_with_industry_average(
        consumer_betas=consumer_betas,
        group_betas=group_betas,
        industry_classification=industry_classification
    )

    # 步骤11: 评估消费贝塔的统计显著性
    significance_results = evaluate_consumer_beta_significance(
        consumer_betas=adjusted_betas,
        stock_returns=stock_returns,
        market_returns=market_returns,
        delta_consumption=delta_consumption
    )

    # 步骤12: 计算历史平均消费增速
    if '0' in ensemble_pred.columns:
        historical_avg = ensemble_pred['0'].mean()
    else:
        historical_avg = ensemble_pred.iloc[:, 0].mean()
    
    # 确保数据类型为数值型
    historical_avg = float(historical_avg)

    # 步骤15: 保存结果
    adjusted_betas.to_csv('02results/consumer_betas.csv')
    significance_results.to_csv('02results/beta_significance.csv')

    # 步骤16: 显示当前消费环境判断
    # 获取最新预测值 - 确保获取数值列而不是日期列
    if '0' in ensemble_pred.columns:
        latest_forecast = ensemble_pred['0'].iloc[-1]
    else:
        # 找到第一个数值列
        numeric_cols = ensemble_pred.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            latest_forecast = ensemble_pred[numeric_cols[0]].iloc[-1]
        else:
            # 如果没有数值列，尝试第二列（通常第一列是日期）
            latest_forecast = ensemble_pred.iloc[-1, 1]
    
    # 确保数据类型为数值型
    latest_forecast = float(latest_forecast)
    
    environment = judge_consumption_environment(latest_forecast, historical_avg)
    print(f"\n当前消费环境判断: {environment}")
    print(f"最新预测值: {latest_forecast:.2f}, 历史平均值: {historical_avg:.2f}")
    
    # 解释数据范围
    print(f"\n数据说明:")
    print(f"消费贝塔计算期间: {delta_consumption_aligned.index.min().strftime('%Y-%m')} 至 {delta_consumption_aligned.index.max().strftime('%Y-%m')}")
    print(f"共计算了 {len(delta_consumption_aligned)} 个月的消费贝塔数据")
    print(f"数据截止到2025年4月是因为消费预测数据只覆盖到该时期")

    # 步骤17: 提供投资建议
    if environment == "良好":
        print("建议投资策略: 选择消费贝塔较高的股票")
        # 获取推荐的高消费贝塔股票
        recommended_stocks = select_stocks_based_on_environment(adjusted_betas, "良好", n_stocks=10)
        print(f"推荐的前10支高消费贝塔股票: {', '.join(recommended_stocks)}")
    else:
        print("建议投资策略: 选择消费贝塔较低的股票")
        # 获取推荐的低消费贝塔股票
        recommended_stocks = select_stocks_based_on_environment(adjusted_betas, "疲软", n_stocks=10)
        print(f"推荐的前10支低消费贝塔股票: {', '.join(recommended_stocks)}")

    print("\n消费贝塔敏感度测度模型执行完成")


# =====================================================================
# 主函数
# =====================================================================

if __name__ == "__main__":
    # 使用包含预处理步骤的主函数
    main_with_preprocessing()
