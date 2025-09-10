# -*- coding: utf-8 -*-
"""
赢家输家分类预测策略
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from datetime import datetime
import warnings
import shutil
import pickle
import time
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# 设置警告过滤
warnings.filterwarnings('ignore')

# ============================= 配置参数 =============================

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# 时间参数
START_DATE = pd.Timestamp('2005-01-01')

# 模型参数
THRESHOLD_WINNER = 0.7  # 赢家分位数阈值，收益率排名前30%的股票
THRESHOLD_LOSER = 0.3   # 输家分位数阈值，收益率排名后30%的股票
LOOK_BACK_WINDOW = 12   # 动量因子回顾期
REVERSAL_WINDOW = 60    # 长期反转因子回顾期
CACHE_DIR = './cache'   # 缓存目录
RESULTS_DIR = './03results'  # 结果保存目录

# 性能优化参数
N_JOBS = -1  # 使用所有可用CPU核心

# ============================= 数据处理模块 =============================

class DataProcessor:
    def __init__(self, data_path):
        """初始化数据处理器"""
        self.data_path = data_path
        self.cache_path = os.path.join(CACHE_DIR, 'processed_data.pkl')
        self.cache_metadata_path = os.path.join(CACHE_DIR, 'data_metadata.pkl')

        # 创建缓存目录
        if not os.path.exists(CACHE_DIR):
            os.makedirs(CACHE_DIR)

    def load_data(self):
        """
        加载数据
        检查源文件是否更新，如果更新则重新加载并更新缓存
        """
        # 输出完整的CSV文件路径和最后修改时间
        abs_path = os.path.abspath(self.data_path)
        print(f"正在加载数据文件: {abs_path}")

        if os.path.exists(abs_path):
            mod_time = datetime.fromtimestamp(os.path.getmtime(abs_path))
            file_size = os.path.getsize(abs_path) / 1024  # KB
            print(f"文件存在 - 大小: {file_size:.2f} KB, 修改时间: {mod_time}")
        else:
            print(f"错误: 文件不存在!")
            return pd.DataFrame()  # 返回空DataFrame

        # 获取CSV文件的最后修改时间
        csv_modified_time = os.path.getmtime(self.data_path) if os.path.exists(self.data_path) else 0

        # 检查缓存元数据
        use_cache = False
        if os.path.exists(self.cache_metadata_path) and os.path.exists(self.cache_path):
            try:
                with open(self.cache_metadata_path, 'rb') as f:
                    cache_metadata = pickle.load(f)
                # 如果源文件未修改，使用缓存
                if csv_modified_time <= cache_metadata.get('modified_time', 0):
                    use_cache = True
            except:
                use_cache = False

        if use_cache:
            print("从缓存加载数据...")
            with open(self.cache_path, 'rb') as f:
                data = pickle.load(f)

            # 输出有关数据的信息
            min_date = data['date'].min()
            max_date = data['date'].max()
            print(f"数据时间范围: {min_date} 至 {max_date}")
            print(f"股票数量: {data['ticker'].nunique()}")

            return data

        print("加载原始数据...")
        # 读取CSV文件，注意这里的格式已经是转置的
        data = pd.read_csv(self.data_path, index_col=0)

        # 检查CSV中的列名（这些应该是日期），加入错误处理
        try:
            # 尝试自动转换日期，但添加错误处理
            csv_dates = pd.to_datetime(data.columns, errors='coerce')
            invalid_dates = csv_dates.isna().sum()
            if invalid_dates > 0:
                print(f"警告: {invalid_dates} 个列名无法解析为日期格式。尝试使用备用格式...")
        except Exception as e:
            print(f"警告: 无法将列名转换为日期: {e}")
            print("尝试替代方法处理日期...")
            # 查看一些列名以分析格式
            print(f"列名样例: {list(data.columns)[:5]}")
            # 可以手动指定日期格式
            try:
                # 尝试解析常见的日期格式
                csv_dates = pd.to_datetime(data.columns, format='%Y%m%d', errors='coerce')
            except:
                # 如果仍然失败，使用原始列名
                csv_dates = data.columns
                print("使用原始列名，不进行日期转换")

        # 输出一些日期相关信息
        if hasattr(csv_dates, 'min') and callable(csv_dates.min):
            print(f"CSV文件中的日期范围: {csv_dates.min()} 至 {csv_dates.max()}")
            print(f"CSV文件中的日期数量: {len(csv_dates)}")
            print(f"最近10个日期: {sorted(csv_dates)[-10:]}")

        # 转换为长格式（每行一个股票-日期的观测值）
        data_long = data.stack().reset_index()
        data_long.columns = ['date', 'ticker', 'monthly_return']  # 保持原有的列名顺序

        # 确保日期格式正确
        data_long['date'] = pd.to_datetime(data_long['date'], errors='coerce')
        invalid_dates = data_long['date'].isna().sum()
        if invalid_dates > 0:
            print(f"警告: {invalid_dates} 个日期值无法解析。")

        # 确保收益率为数值类型
        data_long['monthly_return'] = pd.to_numeric(data_long['monthly_return'], errors='coerce')

        # 输出有关数据的信息
        min_date = data_long['date'].min()
        max_date = data_long['date'].max()
        print(f"数据时间范围: {min_date} 至 {max_date}")
        print(f"股票数量: {data_long['ticker'].nunique()}")

        # 保存到缓存
        with open(self.cache_path, 'wb') as f:
            pickle.dump(data_long, f)

        # 保存元数据
        cache_metadata = {
            'modified_time': csv_modified_time,
            'created_time': time.time(),
            'min_date': min_date,
            'max_date': max_date
        }
        with open(self.cache_metadata_path, 'wb') as f:
            pickle.dump(cache_metadata, f)

        return data_long

    def preprocess_data(self, data):
        """
        数据预处理
        处理缺失值
        """
        print("数据预处理...")

        # 按股票和日期排序
        data = data.sort_values(['ticker', 'date'])

        # 处理缺失值 - 使用论文中的方法：对于停牌等原因导致的缺失数据，使用100%收益率
        data['monthly_return'] = data['monthly_return'].fillna(1.0)  # 100%收益率

        return data

# ============================= 因子计算模块 =============================

class FactorCalculator:
    def __init__(self):
        """初始化因子计算器"""
        self.factors = ['momentum', 'short_reversal', 'long_reversal', 'seasonality']

    def calculate_all_factors(self, data):
        """
        并行计算所有因子
        """
        print("开始计算因子...")
        grouped = data.groupby('ticker')

        # 并行计算每个股票的因子
        results = Parallel(n_jobs=N_JOBS)(
            delayed(self._calculate_stock_factors)(group)
            for _, group in tqdm(grouped, desc="计算因子")
        )

        # 合并结果
        result_df = pd.concat(results, ignore_index=True)

        return result_df

    def _calculate_stock_factors(self, stock_data):
        """计算单个股票的所有因子"""
        stock_data = stock_data.copy()

        # 计算动量因子 - 过去2-12个月的累积收益率
        stock_data['momentum'] = self._calculate_momentum(stock_data)

        # 计算短期反转因子 - 上个月的收益率
        stock_data['short_reversal'] = self._calculate_short_reversal(stock_data)

        # 计算长期反转因子 - 过去13-60个月的累积收益率
        stock_data['long_reversal'] = self._calculate_long_reversal(stock_data)

        # 计算季节性因子 - 过去五年同一月份的累积收益率
        stock_data['seasonality'] = self._calculate_seasonality(stock_data)

        return stock_data

    def _calculate_momentum(self, data):
        """
        计算动量因子
        使用过去2-12个月的累积收益率
        """
        returns = data['monthly_return'].values
        momentum = np.zeros(len(returns))

        for i in range(12, len(returns)):
            # 计算过去第2-12个月的累积收益
            if i >= 12:
                cum_return = 1
                for j in range(2, 13):
                    cum_return *= (1 + returns[i - j])
                momentum[i] = cum_return - 1

        return momentum

    def _calculate_short_reversal(self, data):
        """
        计算短期反转因子
        使用上个月的收益率
        """
        return data['monthly_return'].shift(1)

    def _calculate_long_reversal(self, data):
        """
        计算长期反转因子
        使用过去13-60个月的累积收益率
        """
        returns = data['monthly_return'].values
        long_reversal = np.zeros(len(returns))

        for i in range(60, len(returns)):
            # 计算过去第13-60个月的累积收益
            if i >= 60:
                cum_return = 1
                for j in range(13, 61):
                    cum_return *= (1 + returns[i - j])
                long_reversal[i] = cum_return - 1

        return long_reversal

    def _calculate_seasonality(self, data):
        """
        计算季节性因子
        使用过去五年同一月份的收益率
        """
        returns = data['monthly_return'].values
        seasonality = np.zeros(len(returns))

        for i in range(60, len(returns)):
            # 计算过去5年同一月份的累积收益
            same_month_returns = []
            for j in [12, 24, 36, 48, 60]:
                if i >= j:
                    same_month_returns.append(returns[i - j])

            if same_month_returns:
                cum_return = 1
                for r in same_month_returns:
                    cum_return *= (1 + r)
                seasonality[i] = cum_return - 1

        return seasonality

# ============================= 机器学习模型模块 =============================

class WinnerLoserClassifier:
    def __init__(self):
        """初始化分类器"""
        self.scaler = StandardScaler()
        self.model = LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=42, max_iter=1000)
        self.feature_cols = ['momentum', 'short_reversal', 'long_reversal', 'seasonality']

    def prepare_data(self, data, date):
        """
        准备训练数据
        界定赢家、输家和中性股票
        """
        # 选择特定日期的数据
        current_data = data[data['date'] == date].copy()

        # 计算收益率分位数
        ret_quantile = current_data['monthly_return'].rank(pct=True)

        # 界定赢家、输家和中性
        current_data['class'] = np.where(ret_quantile >= THRESHOLD_WINNER, 1,  # 赢家
                                         np.where(ret_quantile <= THRESHOLD_LOSER, -1,  # 输家
                                                  0))  # 中性

        return current_data

    def train(self, train_data):
        """训练模型"""
        # 删除缺失值
        train_data = train_data.dropna(subset=self.feature_cols + ['class'])

        # 准备特征和标签
        X = train_data[self.feature_cols].values
        y = train_data['class'].values

        # 标准化特征
        X_scaled = self.scaler.fit_transform(X)

        # 训练模型
        self.model.fit(X_scaled, y)

        return self

    def predict_proba(self, test_data):
        """预测概率"""
        # 准备特征
        X = test_data[self.feature_cols].values

        # 标准化特征
        X_scaled = self.scaler.transform(X)

        # 预测概率
        proba = self.model.predict_proba(X_scaled)

        # 获取类别顺序
        classes = self.model.classes_

        # 创建结果DataFrame
        result = test_data.copy()

        # 添加概率列
        for i, cls in enumerate(classes):
            if cls == 1:
                result['winner_prob'] = proba[:, i]
            elif cls == -1:
                result['loser_prob'] = proba[:, i]
            else:
                result['neutral_prob'] = proba[:, i]

        return result

    def classify_stocks(self, data, date):
        """按月对所有股票进行分类，返回每只股票的类别"""
        # 选择特定日期的数据
        current_data = data[data['date'] == date].copy()

        # 计算收益率分位数
        ret_quantile = current_data['monthly_return'].rank(pct=True)

        # 界定赢家、输家和中性
        current_data['class'] = np.where(ret_quantile >= THRESHOLD_WINNER, 1,  # 赢家
                                         np.where(ret_quantile <= THRESHOLD_LOSER, -1,  # 输家
                                                  0))  # 中性

        # 返回包含ticker和class的Series，ticker为索引
        return current_data.set_index('ticker')['class']

# ============================= 回测模块 =============================

class Backtester:
    def __init__(self):
        """初始化回测器"""
        self.returns = []
        self.positions = []
        self.dates = []
        self.classifications = {}  # 新增：存储每月的股票分类结果

    def run_backtest(self, data, model):
        """
        运行回测
        使用滑动窗口方法
        """
        print("开始回测...")

        # 动态确定结束日期为数据中的最大日期
        end_date = data['date'].max()
        print(f"数据结束日期: {end_date}")

        # 获取唯一的月份日期，按月回测
        # 我们需要确保数据是按月组织的
        data['year_month'] = data['date'].dt.to_period('M')
        unique_months = sorted(data['year_month'].unique())
        unique_dates = []

        # 对于每个月，找到该月的最后一个交易日
        for month in unique_months:
            month_data = data[data['year_month'] == month]
            last_date = month_data['date'].max()
            unique_dates.append(last_date)

        unique_dates = sorted(unique_dates)

        print(f"回测期间: {unique_dates[0]} 至 {unique_dates[-1]}")
        print(f"共有 {len(unique_dates)} 个月进行回测")

        # 找到开始日期的索引
        start_index = 0
        for i, date in enumerate(unique_dates):
            if date >= START_DATE and i >= 12:  # 确保有足够的历史数据计算因子（改为12个月）
                start_index = i
                break

        if start_index == 0:
            raise ValueError("没有足够的历史数据进行回测")

        # 滑动窗口回测
        for i in tqdm(range(start_index, len(unique_dates) - 1), desc="回测进度"):
            train_date = unique_dates[i]
            test_date = unique_dates[i + 1]

            # 准备训练数据
            train_data = model.prepare_data(data, train_date)

            # 训练模型
            model.train(train_data)

            # 准备测试数据
            test_data = data[data['date'] == train_date].copy()

            # 预测
            predictions = model.predict_proba(test_data)

            # 计算并保存各股票的分类结果
            classification = model.classify_stocks(data, test_date)
            self.classifications[test_date] = classification

            # 构建投资组合
            portfolio_return = self._build_portfolio(predictions, data, test_date)

            # 记录结果
            self.returns.append(portfolio_return)
            self.dates.append(test_date)

        return self.analyze_results()

    def _build_portfolio(self, predictions, data, next_date):
        """构建投资组合并计算收益"""
        # 选择赢家概率最高的前20%股票
        sorted_preds = predictions.sort_values('winner_prob', ascending=False)
        n_select = max(1, int(len(sorted_preds) * 0.2))
        selected_stocks = sorted_preds.head(n_select)['ticker'].values

        # 获取下期收益
        next_returns = data[data['date'] == next_date].set_index('ticker')['monthly_return']

        # 计算等权重组合收益
        portfolio_returns = []
        for stock in selected_stocks:
            if stock in next_returns.index:
                portfolio_returns.append(next_returns[stock])

        if portfolio_returns:
            return np.mean(portfolio_returns)
        else:
            return 0

    def analyze_results(self):
        """分析回测结果"""
        # 检查是否有回测结果
        if len(self.returns) == 0 or len(self.dates) == 0:
            print("警告: 没有回测结果，请检查数据和参数设置")
            return {
                '年化收益率': 0,
                '夏普比率': 0,
                '最大回撤': 0,
                '胜率': 0
            }
        
        # 计算累积收益
        cum_returns = (1 + pd.Series(self.returns)).cumprod()

        # 计算年化收益
        total_days = (self.dates[-1] - self.dates[0]).days
        years = total_days / 365.0
        annual_return = (cum_returns.iloc[-1] ** (1 / years)) - 1

        # 计算夏普比率
        excess_returns = pd.Series(self.returns) - 0.02 / 12  # 假设无风险利率为2%年化
        sharpe_ratio = np.sqrt(12) * excess_returns.mean() / excess_returns.std()

        # 计算最大回撤
        rolling_max = cum_returns.expanding().max()
        drawdowns = cum_returns / rolling_max - 1
        max_drawdown = drawdowns.min()

        # 绘制结果
        self._plot_results(cum_returns)

        results = {
            '年化收益率': annual_return,
            '夏普比率': sharpe_ratio,
            '最大回撤': max_drawdown,
            '胜率': sum(r > 0 for r in self.returns) / len(self.returns)
        }

        print("\n回测结果:")
        for key, value in results.items():
            print(f"{key}: {value:.4f}")

        return results

    def _plot_results(self, cum_returns):
        """绘制回测结果"""
        plt.figure(figsize=(12, 8))

        # 累积收益曲线
        plt.subplot(2, 1, 1)
        plt.plot(self.dates, cum_returns, label='策略累积收益')
        plt.title('累积收益曲线', fontsize=14)
        plt.legend()
        plt.grid(True)

        # 月度收益分布
        plt.subplot(2, 1, 2)
        sns.histplot(self.returns, bins=50, kde=True)
        plt.title('月度收益分布', fontsize=14)
        plt.xlabel('月度收益')
        plt.ylabel('频率')
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'backtest_results.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def get_classifications(self):
        """返回所有月份的股票分类结果"""
        return self.classifications

    def save_classifications(self, filename="classification_results.csv"):
        """保存股票分类结果到CSV文件"""
        # 将分类结果转换为平面DataFrame
        all_results = []

        for date, classification in self.classifications.items():
            df = classification.reset_index()
            df.columns = ['ticker', 'class']  # 确保列名正确
            df['date'] = date
            all_results.append(df)

        if all_results:
            result_df = pd.concat(all_results)
            # 重新排列列，使日期在前面
            result_df = result_df[['date', 'ticker', 'class']]
            # 透视表：行为日期，列为股票代码，值为分类
            pivot_df = result_df.pivot(index='date', columns='ticker', values='class')
            # 保存到CSV
            output_path = os.path.join(RESULTS_DIR, filename)
            pivot_df.to_csv(output_path)
            print(f"分类结果已保存到 {output_path}")
            return pivot_df
        else:
            print("没有分类结果可保存")
            return None

# ============================= 工具函数模块 =============================

def build_stock_pool(predictions, top_percent=0.2):
    """
    构建股票池
    predictions: 包含winner_prob的DataFrame
    top_percent: 选择赢家概率最高的前多少比例的股票
    """
    # 按赢家概率从高到低排序
    sorted_preds = predictions.sort_values('winner_prob', ascending=False)

    # 选择赢家概率最高的前top_percent的股票
    n_select = max(1, int(len(sorted_preds) * top_percent))
    stock_pool = sorted_preds.head(n_select)

    return stock_pool[['ticker', 'winner_prob']]


def validate_classification(data, classifications, date):
    """验证特定日期的分类结果是否正确"""
    # 获取特定日期的实际收益率
    actual_returns = data[data['date'] == date].set_index('ticker')['monthly_return']

    # 获取特定日期的分类结果
    if date in classifications:
        class_series = classifications[date]

        # 分类计数
        winners = class_series[class_series == 1]
        neutrals = class_series[class_series == 0]
        losers = class_series[class_series == -1]

        # 计算各类股票的平均收益率
        winner_returns = actual_returns.loc[winners.index].mean()
        neutral_returns = actual_returns.loc[neutrals.index].mean()
        loser_returns = actual_returns.loc[losers.index].mean()

        print(f"日期: {date}")
        print(f"赢家数量: {len(winners)}, 平均收益率: {winner_returns:.4f}")
        print(f"中性数量: {len(neutrals)}, 平均收益率: {neutral_returns:.4f}")
        print(f"输家数量: {len(losers)}, 平均收益率: {loser_returns:.4f}")
        print("-" * 30)

        return True
    else:
        print(f"警告: 日期 {date} 的分类结果不存在")
        return False


def plot_classification_performance(data, classifications, dates=None):
    """
    绘制分类表现，展示赢家、中性和输家的平均收益率随时间变化

    参数:
    data: 原始数据，包含date、ticker和monthly_return
    classifications: 分类结果字典，键为日期，值为分类DataFrame
    dates: 要绘制的日期列表，如果为None则使用所有日期
    """
    if dates is None:
        dates = sorted(classifications.keys())

    winner_returns = []
    neutral_returns = []
    loser_returns = []

    for date in dates:
        if date in classifications:
            # 获取特定日期的实际收益率
            actual_returns = data[data['date'] == date].set_index('ticker')['monthly_return']
            class_df = classifications[date]

            # 计算每类的平均收益率
            winners = actual_returns.loc[class_df[class_df == 1].index].dropna()
            neutrals = actual_returns.loc[class_df[class_df == 0].index].dropna()
            losers = actual_returns.loc[class_df[class_df == -1].index].dropna()

            winner_returns.append(winners.mean())
            neutral_returns.append(neutrals.mean())
            loser_returns.append(losers.mean())

    # 绘制结果
    plt.figure(figsize=(12, 6))
    plt.plot(dates, winner_returns, 'g-', label='赢家')
    plt.plot(dates, neutral_returns, 'b-', label='中性')
    plt.plot(dates, loser_returns, 'r-', label='输家')
    plt.title('分类表现随时间变化', fontsize=14)
    plt.xlabel('日期')
    plt.ylabel('平均收益率')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'classification_performance.png'), dpi=300, bbox_inches='tight')
    plt.close()


def analyze_stock_pool(data, model, date, top_percent=0.2):
    """
    分析特定日期的股票池构成

    参数:
    data: 原始数据，包含因子
    model: 训练好的WinnerLoserClassifier模型
    date: 分析日期
    top_percent: 股票池选择比例
    """
    # 准备测试数据
    test_data = data[data['date'] == date].copy()

    # 预测
    predictions = model.predict_proba(test_data)

    # 构建股票池
    stock_pool = build_stock_pool(predictions, top_percent)

    # 分析股票池的因子分布
    factor_cols = ['momentum', 'short_reversal', 'long_reversal', 'seasonality']
    pool_factors = test_data[test_data['ticker'].isin(stock_pool['ticker'])][factor_cols]
    all_factors = test_data[factor_cols]

    # 绘制因子分布对比
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.flatten()

    for i, factor in enumerate(factor_cols):
        sns.kdeplot(all_factors[factor].dropna(), ax=axs[i], label='全市场', color='blue')
        sns.kdeplot(pool_factors[factor].dropna(), ax=axs[i], label='股票池', color='red')
        axs[i].set_title(f'{factor} 分布对比')
        axs[i].legend()
        axs[i].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'stock_pool_factors.png'), dpi=300, bbox_inches='tight')
    plt.close()

    return stock_pool


def compare_pool_performance(data, stock_pools, market_index=None):
    """
    比较不同时期股票池的表现

    参数:
    data: 原始数据，包含date、ticker和monthly_return
    stock_pools: 字典，键为日期，值为该日期的股票池
    market_index: 市场指数收益率序列，如果为None则使用全市场平均收益率
    """
    dates = sorted(stock_pools.keys())
    pool_returns = []
    market_returns = []

    for date in dates:
        # 获取下一期收益率
        next_date = min([d for d in dates if d > date], default=None)
        if next_date is None:
            continue

        # 股票池收益率
        pool_stocks = stock_pools[date]['ticker'].values
        next_returns = data[data['date'] == next_date].set_index('ticker')['monthly_return']
        pool_return = next_returns.loc[pool_stocks].mean()
        pool_returns.append(pool_return)

        # 市场收益率
        if market_index is None:
            market_return = next_returns.mean()
        else:
            market_return = market_index.loc[next_date]
        market_returns.append(market_return)

    # 计算累积收益
    pool_cum_returns = (1 + pd.Series(pool_returns)).cumprod()
    market_cum_returns = (1 + pd.Series(market_returns)).cumprod()

    # 绘制结果
    plt.figure(figsize=(12, 6))
    plt.plot(dates[:-1], pool_cum_returns, 'r-', label='股票池')
    plt.plot(dates[:-1], market_cum_returns, 'b-', label='市场')
    plt.title('股票池与市场累积收益对比', fontsize=14)
    plt.xlabel('日期')
    plt.ylabel('累积收益')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'pool_vs_market.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 计算超额收益和信息比率
    excess_returns = np.array(pool_returns) - np.array(market_returns)
    annual_excess_return = np.mean(excess_returns) * 12
    annual_excess_std = np.std(excess_returns) * np.sqrt(12)
    
    if annual_excess_std > 0:
        information_ratio = annual_excess_return / annual_excess_std
        print(f"年化超额收益: {annual_excess_return:.4f}")
        print(f"信息比率: {information_ratio:.4f}")


# ============================= 主程序 =============================

def main():
    """主函数"""
    import sys
    import io
    
    # 创建字符串缓冲区来捕获输出
    log_buffer = io.StringIO()
    
    # 重定向标准输出
    original_stdout = sys.stdout
    sys.stdout = Tee(original_stdout, log_buffer)
    
    try:
        print("赢家输家分类预测策略 - 重构版本（仅分类）")
        print("=" * 50)
        print(f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # 创建结果目录
        if not os.path.exists(RESULTS_DIR):
            os.makedirs(RESULTS_DIR)
            print(f"创建结果目录: {RESULTS_DIR}")

        # 设置数据路径
        data_path = 'stock_monthly_returns.csv'

        # 检查数据文件是否存在
        if not os.path.exists(data_path):
            print(f"错误: 未找到数据文件 '{data_path}'")
            return

        # 强制清除缓存
        force_clear_cache = True  # 设置为True强制重新加载数据
        if force_clear_cache and os.path.exists(CACHE_DIR):
            print("强制清除缓存...")
            shutil.rmtree(CACHE_DIR)
            os.makedirs(CACHE_DIR)

        # 1. 数据处理
        print("\n1. 数据处理阶段")
        processor = DataProcessor(data_path)
        data = processor.load_data()
        data = processor.preprocess_data(data)

        # 显示数据的时间范围
        print(f"数据时间范围: {data['date'].min()} 至 {data['date'].max()}")
        print(f"共有 {data['ticker'].nunique()} 只股票")

        # 获取数据集中的最新日期
        latest_data_date = data['date'].max()
        date_suffix = latest_data_date.strftime('%Y%m%d')
        print(f"数据集最新日期: {latest_data_date}, 将用于文件命名")

        # 2. 因子计算
        print("\n2. 因子计算阶段")
        factor_calculator = FactorCalculator()
        data_with_factors = factor_calculator.calculate_all_factors(data)

        # 显示因子计算后的数据概况
        print(f"因子计算后的数据: {len(data_with_factors)} 行")

        # 检查因子缺失情况
        factor_cols = ['momentum', 'short_reversal', 'long_reversal', 'seasonality']
        missing_counts = {col: data_with_factors[col].isna().sum() for col in factor_cols}
        print("因子缺失值统计:")
        for col, count in missing_counts.items():
            pct = count / len(data_with_factors) * 100
            print(f"  {col}: {count} ({pct:.2f}%)")

        # 3. 股票分类（不进行回测）
        print("\n3. 股票分类阶段")
        model = WinnerLoserClassifier()
        
        # 获取所有唯一日期
        unique_dates = sorted(data_with_factors['date'].unique())
        print(f"开始对 {len(unique_dates)} 个日期进行股票分类...")
        
        # 存储分类结果
        all_classifications = {}
        
        for date in tqdm(unique_dates, desc="分类进度"):
            try:
                # 对每个日期进行分类
                classification = model.classify_stocks(data_with_factors, date)
                all_classifications[date] = classification
            except Exception as e:
                print(f"日期 {date} 分类失败: {e}")
                continue
        
        print(f"成功分类 {len(all_classifications)} 个日期")

        # 4. 保存分类结果
        print("\n4. 保存分类结果")
        if all_classifications:
            # 转换为DataFrame格式
            all_results = []
            for date, classification in all_classifications.items():
                df = classification.reset_index()
                df.columns = ['ticker', 'class']
                df['date'] = date
                all_results.append(df)
            
            result_df = pd.concat(all_results)
            result_df = result_df[['date', 'ticker', 'class']]
            
            # 透视表：行为日期，列为股票代码，值为分类
            pivot_df = result_df.pivot(index='date', columns='ticker', values='class')
            
            # 保存到CSV
            output_path = os.path.join(RESULTS_DIR, f"classification_results_{date_suffix}.csv")
            pivot_df.to_csv(output_path)
            print(f"分类结果已保存到 {output_path}")
            
            # 显示部分结果
            print("\n分类结果示例 (前5行, 前10列):")
            print(pivot_df.iloc[:5, :10])
            
            # 统计信息
            print(f"\n分类统计:")
            print(f"时间范围: {pivot_df.index.min()} 至 {pivot_df.index.max()}")
            print(f"股票数量: {pivot_df.shape[1]}")
            print(f"交易日数量: {pivot_df.shape[0]}")
            
        else:
            print("没有分类结果可保存")

        print("\n策略执行完成！")
        print(f"分类结果已保存到: {RESULTS_DIR}")
        print("=" * 50)
        
    finally:
        # 恢复标准输出
        sys.stdout = original_stdout
        
        # 保存完整的终端日志
        log_content = log_buffer.getvalue()
        log_path = os.path.join(RESULTS_DIR, f'complete_execution_log_{date_suffix}.txt')
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(log_content)
        print(f"完整执行日志已保存到: {log_path}")


class Tee:
    """用于同时输出到终端和捕获到缓冲区的类"""
    def __init__(self, *files):
        self.files = files
    
    def write(self, text):
        for file in self.files:
            file.write(text)
            file.flush()
    
    def flush(self):
        for file in self.files:
            if hasattr(file, 'flush'):
                file.flush()


if __name__ == "__main__":
    main()
