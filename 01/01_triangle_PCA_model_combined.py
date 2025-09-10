# 导入必要的库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from autogluon.tabular import TabularPredictor
import pickle
import os
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

import sys
import atexit

# 将标准输出和标准错误同时写入终端与日志文件
class StreamTee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            try:
                stream.write(data)
                stream.flush()
            except Exception:
                # 忽略单个流写入失败，保证其他流不受影响
                pass

    def flush(self):
        for stream in self.streams:
            try:
                stream.flush()
            except Exception:
                pass


def setup_tee_logging(log_path: str = os.path.join('results', 'log.txt')):
    """将终端输出同时保存到 results/log.txt。
    返回一个清理函数；也通过 atexit 自动清理。
    """
    try:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        log_file = open(log_path, 'w', encoding='utf-8')
    except Exception as e:
        print(f"无法创建日志文件 {log_path}: {e}")
        return lambda: None

    original_stdout = sys.stdout
    original_stderr = sys.stderr

    sys.stdout = StreamTee(sys.stdout, log_file)
    sys.stderr = StreamTee(sys.stderr, log_file)

    def cleanup():
        try:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
        finally:
            try:
                log_file.flush()
            finally:
                try:
                    log_file.close()
                except Exception:
                    pass

    atexit.register(cleanup)
    print(f"日志记录开始：所有终端输出将保存到 {log_path}")
    return cleanup

# =====================================================================
# 第一部分：数据加载与预处理
# =====================================================================

def load_data():
    """读取各数据文件"""
    try:
        # 读取社会消费品零售总额数据
        retail_sales = pd.read_csv('CME_Mretailsales.csv')

        # 读取消费者信心指数数据
        consumer_confidence = pd.read_csv('CME_Mbcid2.csv')

        # 读取居民消费价格涨幅数据
        consumer_price = pd.read_csv('CME_Mconsumerpriceratio.csv')

        print("数据加载成功")
        return retail_sales, consumer_confidence, consumer_price

    except Exception as e:
        print(f"数据加载出错: {e}")
        return None, None, None


def clean_data(retail_sales, consumer_confidence, consumer_price):
    """数据清洗和格式调整"""
    # 检查并打印数据的前几行，以确定实际格式
    print("零售数据前5行：")
    print(retail_sales.head())
    print("\n消费者信心数据前5行：")
    print(consumer_confidence.head())
    print("\n消费者价格数据前5行：")
    print(consumer_price.head())

    # 根据实际数据格式处理日期字段
    # 零售总额数据
    if 'Month' in retail_sales.columns:
        try:
            retail_sales['Date'] = pd.to_datetime(retail_sales['Month'])
            retail_sales.set_index('Date', inplace=True)
            print("零售总额数据日期处理成功")
        except Exception as e:
            print(f"零售总额数据日期处理失败: {e}")

    # 消费者价格数据
    if 'Month' in consumer_price.columns:
        try:
            consumer_price['Date'] = pd.to_datetime(consumer_price['Month'])
            consumer_price.set_index('Date', inplace=True)
            print("消费者价格数据日期处理成功")
        except Exception as e:
            print(f"消费者价格数据日期处理失败: {e}")

    # 消费者信心数据
    if 'Staper' in consumer_confidence.columns:
        try:
            consumer_confidence['Date'] = pd.to_datetime(consumer_confidence['Staper'])
            consumer_confidence.set_index('Date', inplace=True)
            print("消费者信心数据日期处理成功")
        except Exception as e:
            print(f"消费者信心数据日期处理失败: {e}")
            try:
                consumer_confidence['Date'] = pd.to_datetime(consumer_confidence['Staper'], format='%Y-%m')
                consumer_confidence.set_index('Date', inplace=True)
                print("使用%Y-%m格式成功解析日期")
            except Exception as e2:
                print(f"替代方法也失败: {e2}")

    if 'Datasign' in consumer_price.columns:
        consumer_price = consumer_price[consumer_price['Datasign'] == 'A']
        print("已筛选消费者价格A=本月同比数据")

    # 处理'Datasign'列 - 修改以处理春节波动
    if 'Datasign' in retail_sales.columns:
        a_data = retail_sales[retail_sales['Datasign'] == 'A']
        feb_b_mask = (retail_sales.index.month == 2) & (retail_sales['Datasign'] == 'B')
        feb_b_data = retail_sales[feb_b_mask]

        if not feb_b_data.empty:
            feb_data = feb_b_data.copy()
            feb_data['Datasign'] = 'A'
            jan_data_list = []
            existing_jan_dates = set(a_data[a_data.index.month == 1].index)

            for year, feb_group in feb_b_data.groupby(feb_b_data.index.year):
                for _, feb_row in feb_group.iterrows():
                    jan_row = feb_row.copy()
                    jan_date = pd.Timestamp(year=year, month=1, day=1)
                    jan_row.name = jan_date

                    if jan_date in existing_jan_dates:
                        print(f"注意：跳过已存在的1月数据点 {jan_date}，使用原始数据")
                        continue

                    for col in jan_row.index:
                        if isinstance(jan_row[col], (int, float)) and not pd.isna(jan_row[col]):
                            jan_row[col] = jan_row[col] / 2
                            feb_data.loc[feb_row.name, col] = jan_row[col]

                    jan_row['Datasign'] = 'A'
                    jan_data_list.append(jan_row)

            if jan_data_list:
                jan_data = pd.DataFrame(jan_data_list)
                combined_a_jan = pd.concat([a_data, jan_data])
                combined_a_jan = combined_a_jan[~combined_a_jan.index.duplicated(keep='first')]
                combined_data = pd.concat([combined_a_jan, feb_data])
                combined_data = combined_data[~combined_data.index.duplicated(keep='first')]
                combined_data = combined_data.sort_index()
                retail_sales = combined_data
                print(f"已处理零售总额数据: 保留{len(a_data)}条A=当期数据，处理了{len(feb_data)}条2月B=累计数据，并估算了{len(jan_data)}条1月数据")
            else:
                combined_data = pd.concat([a_data, feb_data])
                combined_data = combined_data.sort_index()
                retail_sales = combined_data
                print(f"已处理零售总额数据: 保留{len(a_data)}条A=当期数据，处理了{len(feb_data)}条2月B=累计数据")
        else:
            retail_sales = a_data
            print(f"已筛选零售总额A=本月同比数据，共{len(a_data)}条")

    # 选择共同的时间范围
    try:
        start_date = max(
            retail_sales.index.min() if isinstance(retail_sales.index, pd.DatetimeIndex) else pd.Timestamp('1900-01-01'),
            consumer_confidence.index.min() if isinstance(consumer_confidence.index, pd.DatetimeIndex) else pd.Timestamp('1900-01-01'),
            consumer_price.index.min() if isinstance(consumer_price.index, pd.DatetimeIndex) else pd.Timestamp('1900-01-01')
        )

        end_date = min(
            retail_sales.index.max() if isinstance(retail_sales.index, pd.DatetimeIndex) else pd.Timestamp('2100-01-01'),
            consumer_confidence.index.max() if isinstance(consumer_confidence.index, pd.DatetimeIndex) else pd.Timestamp('2100-01-01'),
            consumer_price.index.max() if isinstance(consumer_price.index, pd.DatetimeIndex) else pd.Timestamp('2100-01-01')
        )

        print(f"选择的共同时间范围: {start_date} 至 {end_date}")

        if isinstance(retail_sales.index, pd.DatetimeIndex):
            retail_sales = retail_sales.loc[start_date:end_date]
        if isinstance(consumer_confidence.index, pd.DatetimeIndex):
            consumer_confidence = consumer_confidence.loc[start_date:end_date]
        if isinstance(consumer_price.index, pd.DatetimeIndex):
            consumer_price = consumer_price.loc[start_date:end_date]
    except Exception as e:
        print(f"选择共同时间范围时出错: {e}")

    # 确保数据是按照时间顺序排列的
    if isinstance(retail_sales.index, pd.DatetimeIndex):
        retail_sales = retail_sales.sort_index()
    if isinstance(consumer_confidence.index, pd.DatetimeIndex):
        consumer_confidence = consumer_confidence.sort_index()
    if isinstance(consumer_price.index, pd.DatetimeIndex):
        consumer_price = consumer_price.sort_index()

    print(f"数据清洗完成")
    return retail_sales, consumer_confidence, consumer_price


def preprocess_data(retail_sales, consumer_confidence, consumer_price):
    """
    对数据进行预处理，包括：
    1. 缺失值处理
    2. 平稳化转换
    3. Z-score标准化
    """
    # 缺失值处理
    retail_sales = retail_sales.interpolate(method='linear')
    consumer_confidence = consumer_confidence.interpolate(method='linear')
    consumer_price = consumer_price.interpolate(method='linear')

    # 平稳化转换
    if 'Retailsalegryoym' in retail_sales.columns:
        result = adfuller(retail_sales['Retailsalegryoym'].dropna())
        print(f"零售总额同比增长率 ADF检验结果: p值 = {result[1]}")

        if result[1] > 0.05:
            retail_sales['Retailsalegryoym_diff'] = retail_sales['Retailsalegryoym'].diff()
            print("对零售总额同比增长率进行了差分处理")
            result_diff = adfuller(retail_sales['Retailsalegryoym_diff'].dropna())
            print(f"差分后 ADF检验结果: p值 = {result_diff[1]}")

    # Z-score标准化
    scaler = StandardScaler()

    def standardize_data(dataframe):
        numeric_cols = dataframe.select_dtypes(include=[np.number]).columns
        dataframe_numeric = dataframe[numeric_cols]
        dataframe_scaled = pd.DataFrame(
            scaler.fit_transform(dataframe_numeric),
            index=dataframe.index,
            columns=numeric_cols
        )
        return dataframe_scaled

    retail_sales_scaled = standardize_data(retail_sales)
    consumer_confidence_scaled = standardize_data(consumer_confidence)
    consumer_price_scaled = standardize_data(consumer_price)

    print("数据预处理完成")
    return retail_sales_scaled, consumer_confidence_scaled, consumer_price_scaled


# =====================================================================
# 第二部分：主成分分析法（PCA）构建综合因子
# =====================================================================

def build_factors(retail_sales_scaled, consumer_confidence_scaled, consumer_price_scaled):
    """
    使用PCA方法构建三大综合因子:
    1. 总体消费因子
    2. 结构性消费因子
    3. 消费者心理因子
    """
    print("开始构建PCA综合因子")

    def build_factor(data_dict, factor_name):
        merged_data = pd.concat(data_dict.values(), axis=1)
        merged_data.columns = data_dict.keys()
        merged_data = merged_data.dropna()

        pca = PCA()
        pca_result = pca.fit_transform(merged_data)
        factor = pd.Series(pca_result[:, 0], index=merged_data.index, name=factor_name)

        print(f"\n{factor_name}的PCA结果:")
        print(f"特征值: {pca.explained_variance_}")
        print(f"解释方差比例: {pca.explained_variance_ratio_}")
        print(f"累计解释方差: {np.cumsum(pca.explained_variance_ratio_)}")
        print(f"指标载荷: {pca.components_[0]}")

        return factor, pca

    # 总体消费因子
    total_consumption_data = {}
    if 'Retailsale' in retail_sales_scaled.columns:
        total_consumption_data['RetailSale'] = retail_sales_scaled['Retailsale']
    if 'Retailsalegryoym' in retail_sales_scaled.columns:
        total_consumption_data['RetailSaleGrowth'] = retail_sales_scaled['Retailsalegryoym']
    if 'Bcidm0203' in consumer_confidence_scaled.columns:
        total_consumption_data['ConsumerConfidence'] = consumer_confidence_scaled['Bcidm0203']

    total_consumption_factor, pca_total = build_factor(total_consumption_data, "总体消费因子")

    # 结构性消费因子
    structural_consumption_data = {}
    if 'Retailsaletowngryoym' in retail_sales_scaled.columns:
        structural_consumption_data['TownRetailGrowth'] = retail_sales_scaled['Retailsaletowngryoym']
    if 'Retailsalecountrygryoym' in retail_sales_scaled.columns:
        structural_consumption_data['CountryRetailGrowth'] = retail_sales_scaled['Retailsalecountrygryoym']
    if 'Caterretailsalegryoym' in retail_sales_scaled.columns:
        structural_consumption_data['CateringGrowth'] = retail_sales_scaled['Caterretailsalegryoym']
    if 'Commodityretailgryoym' in retail_sales_scaled.columns:
        structural_consumption_data['CommodityGrowth'] = retail_sales_scaled['Commodityretailgryoym']

    structural_consumption_factor, pca_structural = build_factor(structural_consumption_data, "结构性消费因子")

    # 消费者心理因子
    consumer_psychology_data = {}
    if 'Bcidm0201' in consumer_confidence_scaled.columns:
        consumer_psychology_data['ConsumerExpectation'] = consumer_confidence_scaled['Bcidm0201']
    if 'Bcidm0202' in consumer_confidence_scaled.columns:
        consumer_psychology_data['ConsumerSatisfaction'] = consumer_confidence_scaled['Bcidm0202']
    if 'Bcidm0203' in consumer_confidence_scaled.columns:
        consumer_psychology_data['ConsumerConfidence'] = consumer_confidence_scaled['Bcidm0203']

    consumer_psychology_factor, pca_psychology = build_factor(consumer_psychology_data, "消费者心理因子")

    # 可视化三大因子
    plt.close('all')
    plt.figure(figsize=(12, 8), clear=True)
    
    # 设置中文字体显示
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans', 'Liberation Sans']
    plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
    
    # 使用用户偏好的颜色配置
    plt.style.use('seaborn-v0_8')
    colors = plt.cm.RdYlBu_r(np.linspace(0, 1, 3))
    
    plt.subplot(3, 1, 1)
    total_consumption_factor.plot(color=colors[0])
    plt.title('总体消费因子')
    plt.subplot(3, 1, 2)
    structural_consumption_factor.plot(color=colors[1])
    plt.title('结构性消费因子')
    plt.subplot(3, 1, 3)
    consumer_psychology_factor.plot(color=colors[2])
    plt.title('消费者心理因子')
    plt.tight_layout()
    
    # 确保results目录存在
    os.makedirs('results', exist_ok=True)
    # 设置中文字体显示
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans', 'Liberation Sans']
    plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
    
    plt.savefig('results/pca_factors.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 将三大因子合并为一个DataFrame
    factors = pd.concat([
        total_consumption_factor,
        structural_consumption_factor,
        consumer_psychology_factor
    ], axis=1)

    factors.columns = ['C1', 'C2', 'C3']

    print("PCA综合因子构建完成")
    return factors, pca_total, pca_structural, pca_psychology


# =====================================================================
# 第三部分：三角组合模型框架设计
# =====================================================================

def build_triangle_model(retail_sales_scaled, factors, train_test_split=True):
    """
    构建三角组合预测模型：
    1. SARIMA模型——时间结构组件
    2. VAR模型——经济结构组件
    3. AutoML模型——非线性复杂组件
    """
    print("开始构建三角组合预测模型")

    # 确定目标变量
    target_var = None
    if 'Retailsalegryoym' in retail_sales_scaled.columns:
        target_var = retail_sales_scaled['Retailsalegryoym']
    elif 'Retailsalegryoym_diff' in retail_sales_scaled.columns:
        target_var = retail_sales_scaled['Retailsalegryoym_diff']
    else:
        print("错误：找不到目标变量'Retailsalegryoym'")
        return None

    # 确保目标变量和因子有相同的索引
    common_index = target_var.index.intersection(factors.index)
    target_var = target_var.loc[common_index]
    factors = factors.loc[common_index]

    if train_test_split:
        # 划分训练集和测试集（用于模型评估）
        train_size = int(len(target_var) * 0.8)
        train_target = target_var.iloc[:train_size]
        test_target = target_var.iloc[train_size:]
        train_factors = factors.iloc[:train_size]
        test_factors = factors.iloc[train_size:]
    else:
        # 使用全部数据训练（用于最终预测）
        train_target = target_var
        test_target = target_var  # 这里只是占位，不会用于真正的测试
        train_factors = factors
        test_factors = factors

    # SARIMA模型
    def build_sarima_model(train_data, test_data):
        print("\n构建SARIMA模型")
        try:
            if train_data.isnull().any():
                train_data = train_data.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')

            model = sm.tsa.statespace.SARIMAX(
                train_data,
                order=(2, 1, 1),
                seasonal_order=(1, 1, 1, 12),
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            results = model.fit(disp=False)

            forecast = results.get_forecast(steps=len(test_data))
            predicted_mean = forecast.predicted_mean
            predicted_mean.index = test_data.index

            if train_test_split:
                valid_mask = ~(np.isnan(predicted_mean) | np.isnan(test_data))
                if valid_mask.any():
                    rmse = np.sqrt(((predicted_mean[valid_mask] - test_data[valid_mask]) ** 2).mean())
                    print(f"SARIMA模型RMSE: {rmse}")

            return results, predicted_mean

        except Exception as e:
            print(f"SARIMA模型构建失败: {e}")
            dummy_pred = pd.Series(np.nan, index=test_data.index)
            return None, dummy_pred

    # VAR模型
    def build_var_model(train_target, train_factors, test_target):
        print("\n构建VAR模型")
        
        train_data = pd.concat([train_target, train_factors], axis=1)
        train_data.columns = ['target'] + list(train_factors.columns)

        if train_data.isnull().any().any():
            train_data = train_data.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')

        try:
            scaler = StandardScaler()
            train_data_scaled = pd.DataFrame(
                scaler.fit_transform(train_data),
                index=train_data.index,
                columns=train_data.columns
            )

            model = sm.tsa.VAR(train_data_scaled)
            try:
                results = model.fit(maxlags=2)
            except:
                try:
                    results = model.fit(maxlags=1)
                except:
                    simple_data = train_data_scaled[['target', 'C1']]
                    simple_model = sm.tsa.VAR(simple_data)
                    results = simple_model.fit(maxlags=1)

            lag_order = results.k_ar
            input_data = train_data_scaled.values[-lag_order:]
            forecast = results.forecast(input_data, steps=len(test_target))
            forecast_scaled = forecast[:, 0]
            
            if 'simple_model' in locals():
                target_mean = train_data['target'].mean()
                target_std = train_data['target'].std()
                forecast_unscaled = forecast_scaled * target_std + target_mean
            else:
                forecast_unscaled = forecast_scaled * scaler.scale_[0] + scaler.mean_[0]

            predicted_mean = pd.Series(forecast_unscaled, index=test_target.index)

            if train_test_split:
                rmse = np.sqrt(((predicted_mean - test_target) ** 2).mean())
                print(f"VAR模型RMSE: {rmse}")

            return results, predicted_mean

        except Exception as e:
            print(f"VAR模型构建失败: {e}")
            dummy_pred = pd.Series(np.nan, index=test_target.index)
            return None, dummy_pred

    # AutoML模型
    def build_automl_model(train_target, train_factors, test_target, test_factors):
        print("\n构建AutoML模型")
        try:
            train_data = train_factors.copy()
            train_data['target'] = train_target.values

            save_path = 'results/automl_models'
            os.makedirs(save_path, exist_ok=True)

            predictor = TabularPredictor(
                label='target',
                path=save_path,
                eval_metric='root_mean_squared_error'
            )

            predictor.fit(
                train_data=train_data,
                time_limit=300,
                presets='medium_quality'
            )

            if train_test_split:
                test_data = test_factors.copy()
                test_data['target'] = test_target.values
                print("\nAutoML模型性能评估:")
                leaderboard = predictor.leaderboard(test_data)
                print(leaderboard)

            predicted_mean = pd.Series(
                predictor.predict(test_factors),
                index=test_target.index
            )

            if train_test_split:
                rmse = np.sqrt(((predicted_mean - test_target) ** 2).mean())
                print(f"AutoML模型RMSE: {rmse}")

            return predictor, predicted_mean

        except Exception as e:
            print(f"AutoML模型构建失败: {e}")
            return None, pd.Series(np.nan, index=test_target.index)

    # 构建三个基础模型
    sarima_model, sarima_pred = build_sarima_model(train_target, test_target)
    var_model, var_pred = build_var_model(train_target, train_factors, test_target)
    automl_model, automl_pred = build_automl_model(train_target, train_factors, test_target, test_factors)

    # 动态集成方法
    def ensemble_predictions(sarima_pred, var_pred, automl_pred, test_target):
        print("\n构建动态集成模型")

        if sarima_pred is None and var_pred is None and automl_pred is None:
            print("所有模型均构建失败，无法进行集成")
            return None, None

        # 创建有效预测的字典
        valid_preds = {}
        if sarima_pred is not None and not sarima_pred.isnull().all():
            valid_preds['SARIMA'] = sarima_pred
        if var_pred is not None and not var_pred.isnull().all():
            valid_preds['VAR'] = var_pred
        if automl_pred is not None and not automl_pred.isnull().all():
            valid_preds['AutoML'] = automl_pred

        if len(valid_preds) < 2:
            if len(valid_preds) == 1:
                model_name = list(valid_preds.keys())[0]
                print(f"返回{model_name}模型的预测作为最终结果")
                weights_df = pd.DataFrame({model_name: [1.0]}, index=[0])
                return valid_preds[model_name], weights_df
            else:
                return None, None

        # 确保所有预测都有相同的索引
        common_index = None
        for pred in valid_preds.values():
            if common_index is None:
                common_index = pred.index
            else:
                common_index = common_index.intersection(pred.index)

        if common_index.empty:
            print("预测值没有共同的时间索引，无法进行集成")
            return None, None

        # 调整索引
        for name in valid_preds:
            valid_preds[name] = valid_preds[name].loc[common_index]

        if test_target is not None and train_test_split:
            test_target = test_target.loc[common_index.intersection(test_target.index)]

        # 计算权重
        window_size = min(12, len(common_index) // 3)
        if window_size < 2:
            window_size = 2

        weights = pd.DataFrame(index=common_index)
        for name in valid_preds:
            weights[name] = np.nan

        if test_target is not None and not test_target.empty and train_test_split:
            mse_dict = {}
            for name, pred in valid_preds.items():
                valid_idx = pred.index.intersection(test_target.index)
                if not valid_idx.empty:
                    aligned_pred = pred.loc[valid_idx]
                    aligned_target = test_target.loc[valid_idx]
                    mse = ((aligned_pred - aligned_target) ** 2).rolling(window=window_size).mean()
                    mse_dict[name] = mse

            lambda_param = 5

            def calculate_weights(mse_values):
                exp_weights = {name: np.exp(-lambda_param * mse) for name, mse in mse_values.items()}
                total_weight = sum(exp_weights.values())
                if total_weight > 0:
                    return {name: w / total_weight for name, w in exp_weights.items()}
                else:
                    return {name: 1.0 / len(mse_values) for name in mse_values}

            for i in range(window_size, len(common_index)):
                current_idx = common_index[i]
                if current_idx in test_target.index:
                    current_mse = {name: mse.loc[current_idx] if current_idx in mse.index else np.nan
                                   for name, mse in mse_dict.items()}

                    valid_mse = {name: mse for name, mse in current_mse.items()
                                 if not (np.isnan(mse) or np.isinf(mse))}

                    if valid_mse:
                        current_weights = calculate_weights(valid_mse)
                        for name, w in current_weights.items():
                            weights.loc[current_idx, name] = w

            equal_weight = 1.0 / len(valid_preds)
            for name in valid_preds:
                weights.iloc[:window_size, weights.columns.get_loc(name)] = equal_weight
        else:
            equal_weight = 1.0 / len(valid_preds)
            for name in valid_preds:
                weights[name] = equal_weight

        weights = weights.fillna(method='ffill').fillna(method='bfill')
        if weights.isnull().any().any():
            equal_weight = 1.0 / len(valid_preds)
            weights = weights.fillna(equal_weight)

        # 计算集成预测
        ensemble_pred = pd.Series(0.0, index=common_index)
        for name, pred in valid_preds.items():
            ensemble_pred += weights[name] * pred

        if test_target is not None and not test_target.empty and train_test_split:
            common_eval_idx = ensemble_pred.index.intersection(test_target.index)
            if not common_eval_idx.empty:
                rmse = np.sqrt(((ensemble_pred.loc[common_eval_idx] - test_target.loc[common_eval_idx]) ** 2).mean())
                print(f"集成模型RMSE: {rmse}")

        # 可视化预测结果
        if test_target is not None and not test_target.empty and train_test_split:
            plt.close('all')
            plt.figure(figsize=(14, 8), clear=True)
            
            # 设置中文字体显示
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans', 'Liberation Sans']
            plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
            
            # 使用用户偏好的配色方案
            colors = plt.cm.RdYlBu_r(np.linspace(0, 1, len(valid_preds) + 2))
            
            common_vis_idx = test_target.index.intersection(common_index)
            if not common_vis_idx.empty:
                plt.plot(common_vis_idx, test_target.loc[common_vis_idx], 
                        label='实际值', color='black', linewidth=2)

            for i, (name, pred) in enumerate(valid_preds.items()):
                plt.plot(common_index, pred, label=name, 
                        linestyle='--', color=colors[i])
            
            plt.plot(common_index, ensemble_pred, label='集成模型', 
                    linewidth=2, color=colors[-1])
            plt.legend()
            plt.title('各模型预测结果对比')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # 确保results目录存在
            os.makedirs('results', exist_ok=True)
            plt.savefig('results/predictions_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()

            # 可视化权重变化
            plt.figure(figsize=(14, 6), clear=True)
            for i, name in enumerate(valid_preds):
                plt.plot(weights.index, weights[name], label=name, color=colors[i])
            plt.legend()
            plt.title('动态权重变化')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            plt.savefig('results/weights_evolution.png', dpi=300, bbox_inches='tight')
            plt.close()

        return ensemble_pred, weights

    # 集成预测
    ensemble_pred, weights = ensemble_predictions(sarima_pred, var_pred, automl_pred, test_target)

    # 模型评估
    def evaluate_models(test_target, predictions):
        print("\n模型评估")
        
        if not train_test_split:
            print("未进行训练测试分割，跳过评估")
            return None

        metrics = pd.DataFrame(index=['RMSE', 'MAE', 'MAPE', 'DA'],
                              columns=['SARIMA', 'VAR', 'AutoML', 'Ensemble'])

        for model_name, pred in predictions.items():
            if pred is not None:
                common_idx = test_target.index.intersection(pred.index)
                if len(common_idx) > 0:
                    aligned_pred = pred.loc[common_idx]
                    aligned_target = test_target.loc[common_idx]
                    valid_mask = ~(np.isnan(aligned_pred) | np.isnan(aligned_target))

                    if valid_mask.any():
                        # RMSE
                        rmse = np.sqrt(((aligned_pred[valid_mask] - aligned_target[valid_mask]) ** 2).mean())
                        # MAE
                        mae = np.abs(aligned_pred[valid_mask] - aligned_target[valid_mask]).mean()
                        # MAPE
                        mape_mask = (valid_mask) & (aligned_target != 0)
                        if mape_mask.any():
                            mape = np.abs((aligned_pred[mape_mask] - aligned_target[mape_mask]) / aligned_target[mape_mask]).mean() * 100
                        else:
                            mape = np.nan

                        # 方向准确率
                        aligned_pred_diff = aligned_pred[valid_mask].diff().dropna()
                        aligned_target_diff = aligned_target[valid_mask].diff().dropna()
                        common_diff_idx = aligned_pred_diff.index.intersection(aligned_target_diff.index)

                        if len(common_diff_idx) > 0:
                            final_pred_diff = aligned_pred_diff.loc[common_diff_idx]
                            final_target_diff = aligned_target_diff.loc[common_diff_idx]
                            sign_correct = (np.sign(final_pred_diff) == np.sign(final_target_diff))
                            da = sign_correct.mean()
                        else:
                            da = np.nan

                        metrics.loc['RMSE', model_name] = rmse
                        metrics.loc['MAE', model_name] = mae
                        metrics.loc['MAPE', model_name] = mape
                        metrics.loc['DA', model_name] = da

        print(metrics)
        return metrics

    predictions = {
        'SARIMA': sarima_pred,
        'VAR': var_pred,
        'AutoML': automl_pred,
        'Ensemble': ensemble_pred
    }

    metrics = evaluate_models(test_target, predictions)

    # 返回模型和预测结果
    models = {
        'sarima': sarima_model,
        'var': var_model,
        'automl': automl_model
    }

    return models, predictions, weights, metrics


# =====================================================================
# 第四部分：未来消费预测
# =====================================================================

def forecast_future(retail_sales_scaled, factors, models, months_ahead=4):
    """
    使用训练好的模型预测未来几个月的消费情况
    """
    print(f"\n开始预测未来{months_ahead}个月的消费情况...")
    
    # 确定目标变量
    target_var = None
    if 'Retailsalegryoym' in retail_sales_scaled.columns:
        target_var = retail_sales_scaled['Retailsalegryoym']
    elif 'Retailsalegryoym_diff' in retail_sales_scaled.columns:
        target_var = retail_sales_scaled['Retailsalegryoym_diff']
    else:
        print("错误：找不到目标变量'Retailsalegryoym'")
        return None
        
    # 确保目标变量和因子有相同的索引
    common_index = target_var.index.intersection(factors.index)
    target_var = target_var.loc[common_index]
    factors = factors.loc[common_index]
    
    # 创建未来的日期索引
    last_date = target_var.index[-1]
    future_dates = [last_date + pd.DateOffset(months=i+1) for i in range(months_ahead)]
    
    # 预测结果存储
    future_predictions = pd.DataFrame(index=future_dates, 
                                     columns=['SARIMA', 'VAR', 'AutoML', 'Ensemble'])
    
    # 1. SARIMA模型预测
    sarima_model = models.get('sarima')
    if sarima_model is not None:
        try:
            forecast = sarima_model.get_forecast(steps=months_ahead)
            predicted_mean = forecast.predicted_mean
            
            for i, date in enumerate(future_dates):
                if i < len(predicted_mean):
                    future_predictions.loc[date, 'SARIMA'] = predicted_mean[i]
                    
            print(f"SARIMA模型成功预测未来{months_ahead}个月")
        except Exception as e:
            print(f"SARIMA模型预测失败: {e}")
    
    # 2. VAR模型预测
    var_model = models.get('var')
    if var_model is not None:
        try:
            lag_order = var_model.k_ar
            train_data = pd.concat([target_var, factors], axis=1)
            train_data.columns = ['target'] + list(factors.columns)
            
            scaler = StandardScaler()
            train_data_scaled = pd.DataFrame(
                scaler.fit_transform(train_data),
                index=train_data.index,
                columns=train_data.columns
            )
            
            input_data = train_data_scaled.values[-lag_order:]
            forecast = var_model.forecast(input_data, steps=months_ahead)
            forecast_scaled = forecast[:, 0]
            forecast_unscaled = forecast_scaled * scaler.scale_[0] + scaler.mean_[0]
            
            for i, date in enumerate(future_dates):
                if i < len(forecast_unscaled):
                    future_predictions.loc[date, 'VAR'] = forecast_unscaled[i]
            
            print(f"VAR模型成功预测未来{months_ahead}个月")
        except Exception as e:
            print(f"VAR模型预测失败: {e}")
    
    # 3. AutoML模型预测
    automl_model = models.get('automl')
    if automl_model is not None:
        try:
            # 生成未来因子预测（使用趋势外推）
            n = min(6, len(factors))
            recent_factors = factors.iloc[-n:]
            factor_change = (recent_factors.iloc[-1] - recent_factors.iloc[0]) / n
            
            future_factors = pd.DataFrame(index=future_dates, columns=factors.columns)
            for i, date in enumerate(future_dates):
                for col in factors.columns:
                    future_factors.loc[date, col] = factors.iloc[-1][col] + factor_change[col] * (i+1)
            
            future_predictions_automl = automl_model.predict(future_factors)
            
            for i, date in enumerate(future_dates):
                future_predictions.loc[date, 'AutoML'] = future_predictions_automl[i]
            
            print(f"AutoML模型成功预测未来{months_ahead}个月")
        except Exception as e:
            print(f"AutoML模型预测失败: {e}")
    
    # 4. 集成预测
    for date in future_dates:
        valid_preds = future_predictions.loc[date].dropna()
        if not valid_preds.empty:
            future_predictions.loc[date, 'Ensemble'] = valid_preds.mean()
    
    # 5. 消费环境判断
    history_window = min(36, len(target_var))
    historical_avg = target_var.iloc[-history_window:].mean()
    
    future_predictions['Environment'] = future_predictions['Ensemble'].apply(
        lambda x: "良好" if x > historical_avg else "疲软")
    future_predictions['Historical_Avg'] = historical_avg
    
    # 6. 可视化未来预测结果
    plt.close('all')
    plt.figure(figsize=(14, 8), clear=True)
    
    # 设置中文字体显示
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans', 'Liberation Sans']
    plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
    
    # 使用用户偏好的配色方案
    colors = plt.cm.RdYlBu_r(np.linspace(0, 1, 5))
    
    # 绘制历史数据
    history_plot_window = min(24, len(target_var))
    historical_data = target_var.iloc[-history_plot_window:]
    plt.plot(historical_data.index, historical_data, 
             label='历史数据', color='black', marker='o', linewidth=2)
    
    # 绘制预测结果
    model_colors = {'SARIMA': colors[0], 'VAR': colors[1], 'AutoML': colors[2], 'Ensemble': colors[3]}
    for model in ['SARIMA', 'VAR', 'AutoML']:
        if not future_predictions[model].isnull().all():
            plt.plot(future_predictions.index, future_predictions[model], 
                     label=f'{model}预测', linestyle='--', 
                     color=model_colors[model], marker='x')
    
    # 绘制集成预测
    if not future_predictions['Ensemble'].isnull().all():
        plt.plot(future_predictions.index, future_predictions['Ensemble'], 
                 label='集成预测', color=model_colors['Ensemble'], 
                 linewidth=3, marker='s')
    
    # 添加历史平均线
    plt.axhline(y=historical_avg, color='gray', linestyle='-.', 
                label=f'历史平均值: {historical_avg:.2f}')
    
    plt.title(f'消费增长预测 (未来{months_ahead}个月)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 确保results目录存在
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/future_predictions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"未来{months_ahead}个月预测完成")
    return future_predictions


# =====================================================================
# 第五部分：历史滚动窗口预测
# =====================================================================

def historical_rolling_prediction(retail_sales_scaled, consumer_confidence_scaled, consumer_price_scaled):
    """
    进行历史滚动窗口预测：站在每个历史时间点，预测下一个月的消费情况
    """
    print("\n开始进行历史滚动窗口预测...")
    
    # 确定目标变量
    target_var = None
    if 'Retailsalegryoym' in retail_sales_scaled.columns:
        target_var = retail_sales_scaled['Retailsalegryoym']
    elif 'Retailsalegryoym_diff' in retail_sales_scaled.columns:
        target_var = retail_sales_scaled['Retailsalegryoym_diff']
    else:
        print("错误：找不到目标变量'Retailsalegryoym'")
        return None
    
    # 确定最小训练数据量
    min_train_periods = 24
    
    if len(target_var) <= min_train_periods + 1:
        print(f"错误：数据量不足，至少需要{min_train_periods + 1}个观测值")
        return None
    
    # 存储所有预测结果
    all_predictions = []
    all_dates = target_var.index
    start_pred_idx = min_train_periods
    end_pred_idx = len(all_dates) - 1
    
    print(f"数据共有{len(all_dates)}个月，将从第{start_pred_idx+1}个月开始预测")
    
    # 对每个时间点进行预测
    for t in range(start_pred_idx, end_pred_idx):
        current_date = all_dates[t]
        next_date = all_dates[t+1]
        
        print(f"\n站在{current_date.strftime('%Y-%m')}预测{next_date.strftime('%Y-%m')}的消费情况")
        
        # 截取到当前时间点的数据
        current_retail = retail_sales_scaled.loc[:current_date]
        current_consumer_confidence = consumer_confidence_scaled.loc[:current_date]
        current_consumer_price = consumer_price_scaled.loc[:current_date]
        
        try:
            # 构建PCA因子
            factors, _, _, _ = build_factors(
                current_retail, current_consumer_confidence, current_consumer_price
            )
            
            # 确定目标变量
            current_target = None
            if 'Retailsalegryoym' in current_retail.columns:
                current_target = current_retail['Retailsalegryoym']
            elif 'Retailsalegryoym_diff' in current_retail.columns:
                current_target = current_retail['Retailsalegryoym_diff']
            
            # 确保目标变量和因子有相同的索引
            common_index = current_target.index.intersection(factors.index)
            current_target = current_target.loc[common_index]
            current_factors = factors.loc[common_index]
            
            # 简化版的SARIMA模型
            try:
                sarima_model = sm.tsa.statespace.SARIMAX(
                    current_target,
                    order=(2, 1, 1),
                    seasonal_order=(1, 1, 1, 12),
                    enforce_stationarity=False,
                    enforce_invertibility=False
                ).fit(disp=False)
                
                forecast = sarima_model.get_forecast(steps=1)
                sarima_pred = forecast.predicted_mean[0]
            except Exception as e:
                print(f"SARIMA预测失败: {e}")
                sarima_pred = None
            
            # 简化版的VAR模型
            try:
                train_data = pd.concat([current_target, current_factors], axis=1)
                train_data.columns = ['target'] + list(current_factors.columns)
                
                scaler = StandardScaler()
                train_data_scaled = pd.DataFrame(
                    scaler.fit_transform(train_data),
                    index=train_data.index,
                    columns=train_data.columns
                )
                
                var_model = sm.tsa.VAR(train_data_scaled).fit(maxlags=1)
                
                lag_order = var_model.k_ar
                input_data = train_data_scaled.values[-lag_order:]
                forecast = var_model.forecast(input_data, steps=1)
                forecast_scaled = forecast[0, 0]
                var_pred = forecast_scaled * scaler.scale_[0] + scaler.mean_[0]
            except Exception as e:
                print(f"VAR预测失败: {e}")
                var_pred = None
            
            # 计算平均预测
            valid_preds = []
            if sarima_pred is not None and not np.isnan(sarima_pred):
                valid_preds.append(sarima_pred)
            if var_pred is not None and not np.isnan(var_pred):
                valid_preds.append(var_pred)
            
            if valid_preds:
                next_month_pred = np.mean(valid_preds)
            else:
                next_month_pred = current_target.mean()
            
            # 消费环境判断
            history_window = min(36, len(current_target))
            historical_avg = current_target.iloc[-history_window:].mean()
            environment = "良好" if next_month_pred > historical_avg else "疲软"
            
            # 记录预测结果
            prediction = {
                'date': next_date.strftime('%Y/%m/%d'),
                'environment': environment,
                'prediction': next_month_pred,
                'historical_avg': historical_avg,
                'actual': target_var.loc[next_date] if next_date in target_var.index else None
            }
            
            all_predictions.append(prediction)
            
        except Exception as e:
            print(f"在{current_date}预测时发生错误: {e}")
    
    # 转换为DataFrame并保存
    if all_predictions:
        all_predictions_df = pd.DataFrame(all_predictions)
        
        # 计算准确率
        if 'actual' in all_predictions_df.columns and not all_predictions_df['actual'].isnull().all():
            correct_predictions = sum((all_predictions_df['prediction'] > all_predictions_df['historical_avg']) == 
                                      (all_predictions_df['actual'] > all_predictions_df['historical_avg']))
            accuracy = correct_predictions / sum(~all_predictions_df['actual'].isnull())
            print(f"\n环境判断准确率: {accuracy:.2%}")
            
            valid_mask = ~all_predictions_df['actual'].isnull()
            mae = np.abs(all_predictions_df.loc[valid_mask, 'prediction'] - all_predictions_df.loc[valid_mask, 'actual']).mean()
            rmse = np.sqrt(((all_predictions_df.loc[valid_mask, 'prediction'] - all_predictions_df.loc[valid_mask, 'actual']) ** 2).mean())
            print(f"平均绝对误差(MAE): {mae:.4f}")
            print(f"均方根误差(RMSE): {rmse:.4f}")
            
            all_predictions_df['prediction_error'] = all_predictions_df['prediction'] - all_predictions_df['actual']
        
        print(f"\n历史滚动窗口预测完成，共完成{len(all_predictions_df)}个月的预测")
        return all_predictions_df
    else:
        print("没有生成任何预测结果")
        return None


# =====================================================================
# 第六部分：保存结果函数
# =====================================================================

def save_results(factors, models, predictions, weights, metrics, future_predictions=None, historical_predictions=None):
    """保存所有重要变量和结果"""
    print("\n正在保存重要变量...")

    # 创建保存目录（相当于 mkdir -p results）
    os.makedirs('results', exist_ok=True)

    # 保存因子数据
    factors.to_csv('results/pca_factors.csv')
    print("已保存: pca_factors.csv")

    # 保存单独的预测结果
    for name, pred in predictions.items():
        if pred is not None:
            pred.to_csv(f'results/predictions_{name}.csv')
            print(f"已保存: predictions_{name}.csv")

    # 保存权重
    if weights is not None:
        weights.to_csv('results/ensemble_weights.csv')
        print("已保存: ensemble_weights.csv")

    # 保存评估指标
    if metrics is not None:
        metrics.to_csv('results/evaluation_metrics.csv')
        print("已保存: evaluation_metrics.csv")

    # 保存未来预测结果
    if future_predictions is not None:
        future_predictions.to_csv('results/future_predictions_detailed.csv')
        print("已保存: future_predictions_detailed.csv")

    # 保存历史滚动预测结果
    if historical_predictions is not None:
        historical_predictions.to_csv('results/environment_predictions_historical_detailed.csv', index=False)
        print("已保存: environment_predictions_historical_detailed.csv")

    # 尝试保存模型
    try:
        with open('results/sarima_model.pkl', 'wb') as f:
            pickle.dump(models['sarima'], f)
        print("已保存: sarima_model.pkl")
    except:
        print("SARIMA模型无法使用pickle保存")

    try:
        with open('results/var_model.pkl', 'wb') as f:
            pickle.dump(models['var'], f)
        print("已保存: var_model.pkl")
    except:
        print("VAR模型无法使用pickle保存")

    print("变量保存完成，保存路径: 'results/'目录")


def generate_investment_advice(future_predictions):
    """根据未来预测的消费环境，生成投资建议"""
    print("\n生成投资建议...")
    
    report = {
        "生成时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "预测范围": f"{future_predictions.index[0].strftime('%Y-%m-%d')} 至 {future_predictions.index[-1].strftime('%Y-%m-%d')}",
        "总体消费环境": "",
        "月度预测": [],
        "主要发现": [],
        "投资建议": []
    }
    
    # 判断总体消费环境趋势
    env_counts = future_predictions['Environment'].value_counts()
    if len(env_counts) > 0:
        most_common_env = env_counts.index[0]
        report["总体消费环境"] = most_common_env
    else:
        report["总体消费环境"] = "无法确定"
    
    # 生成月度预测概要
    for date, row in future_predictions.iterrows():
        month_report = {
            "日期": date.strftime("%Y-%m"),
            "预测值": f"{row['Ensemble']:.4f}",
            "环境判断": row['Environment'],
            "历史平均值": f"{row['Historical_Avg']:.4f}"
        }
        report["月度预测"].append(month_report)
    
    # 生成投资建议
    if report["总体消费环境"] == "良好":
        report["投资建议"].append("采用高贝塔(β)投资策略，增加对消费相关行业的配置")
        report["投资建议"].append("关注具有高增长潜力的消费板块，例如：可选消费、高端消费品等")
        report["投资建议"].append("减少对防御性行业的配置比例")
    else:
        report["投资建议"].append("采用低贝塔(β)投资策略，减少对周期性消费行业的敞口")
        report["投资建议"].append("增加必需消费品、医疗保健等防御性行业的配置")
        report["投资建议"].append("关注具有稳定现金流和高股息的消费类公司")
    
    # 保存报告到文件
    with open('results/investment_advice.txt', 'w', encoding='utf-8') as f:
        f.write("========== 消费环境与投资建议报告 ==========\n")
        f.write(f"生成时间: {report['生成时间']}\n")
        f.write(f"预测范围: {report['预测范围']}\n")
        f.write(f"总体消费环境: {report['总体消费环境']}\n\n")
        
        f.write("月度预测:\n")
        for month in report["月度预测"]:
            f.write(f"  {month['日期']}: 预测值 {month['预测值']}, 环境 {month['环境判断']}\n")
        
        f.write("\n投资建议:\n")
        for advice in report["投资建议"]:
            f.write(f"  * {advice}\n")
        f.write("=============================================\n")
    
    print("投资建议报告已保存至'results/investment_advice.txt'")
    return report


# =====================================================================
# 主函数
# =====================================================================

def main():
    """
    主函数：执行完整的预测流程
    """
    setup_tee_logging()
    print("开始执行消费预测模型")
    
    # 1. 加载数据
    retail_sales, consumer_confidence, consumer_price = load_data()

    if retail_sales is None or consumer_confidence is None or consumer_price is None:
        print("数据加载失败，程序终止")
        return

    # 2. 数据清洗
    retail_sales, consumer_confidence, consumer_price = clean_data(
        retail_sales, consumer_confidence, consumer_price
    )

    # 3. 数据预处理
    retail_sales_scaled, consumer_confidence_scaled, consumer_price_scaled = preprocess_data(
        retail_sales, consumer_confidence, consumer_price
    )

    # 4. 构建PCA综合因子
    factors, pca_total, pca_structural, pca_psychology = build_factors(
        retail_sales_scaled, consumer_confidence_scaled, consumer_price_scaled
    )

    # 5. 进行历史滚动窗口预测
    print("\n=== 开始进行历史滚动窗口预测 ===")
    historical_predictions = historical_rolling_prediction(
        retail_sales_scaled, consumer_confidence_scaled, consumer_price_scaled
    )
    
    # 6. 构建三角组合预测模型（使用训练测试分割进行评估）
    print("\n=== 构建三角组合预测模型（训练测试分割）===")
    models_eval, predictions_eval, weights_eval, metrics_eval = build_triangle_model(
        retail_sales_scaled, factors, train_test_split=True
    )
    
    # 7. 使用全部数据重新训练模型进行未来预测
    print("\n=== 使用全部历史数据训练最终模型 ===")
    models_final, predictions_final, weights_final, metrics_final = build_triangle_model(
        retail_sales_scaled, factors, train_test_split=False
    )
    
    # 8. 预测未来4个月的消费情况
    print("\n=== 开始预测未来消费情况 ===")
    future_predictions = forecast_future(retail_sales_scaled, factors, models_final, months_ahead=4)
    
    # 9. 生成投资建议
    if future_predictions is not None and not future_predictions.empty:
        print("\n=== 生成投资建议 ===")
        investment_advice = generate_investment_advice(future_predictions)
        
        # 输出预测结果摘要
        next_month_date = future_predictions.index[0]
        next_month_pred = future_predictions.loc[next_month_date, 'Ensemble']
        next_month_env = future_predictions.loc[next_month_date, 'Environment']
        historical_avg = future_predictions.loc[next_month_date, 'Historical_Avg']
        
        print(f"\n下个月消费环境判断: {next_month_env}")
        print(f"最新预测值: {next_month_pred:.4f}, 历史平均值: {historical_avg:.4f}")
        print(f"建议投资策略: {'高贝塔' if next_month_env == '良好' else '低贝塔'}")
    
    # 10. 保存所有结果
    print("\n=== 保存结果 ===")
    save_results(
        factors=factors,
        models=models_eval,  # 保存评估模型的结果
        predictions=predictions_eval,
        weights=weights_eval,
        metrics=metrics_eval,
        future_predictions=future_predictions,
        historical_predictions=historical_predictions
    )
    
    print("\n程序执行完成")
    print("\n生成的文件列表:")
    print("- pca_factors.csv")
    print("- predictions_SARIMA.csv")
    print("- predictions_VAR.csv") 
    print("- predictions_AutoML.csv")
    print("- predictions_Ensemble.csv")
    print("- ensemble_weights.csv")
    print("- environment_predictions_historical_detailed.csv")
    print("- future_predictions_detailed.csv")
    print("- evaluation_metrics.csv")
    print("- investment_advice.txt")


# 运行主函数
if __name__ == "__main__":
    main()