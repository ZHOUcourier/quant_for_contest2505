from mindgo_api import *
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

def init(context):
    set_benchmark('000300.SH')
    
    # 设置交易成本
    set_commission(PerShare(type='stock', cost=0.0002))  # 设置股票每笔交易的手续费为万分之二
    set_slippage(PriceSlippage(0.002))  # 设置滑点0.2%
    
    # 设置日级最大成交比例25%,分钟级最大成交比例50%
    set_volume_limit(0.25, 0.5)
    
    # 初始化RSI+SuperTrend策略参数
    context.rsi_period = 14
    context.rsi_overbought = 70
    context.rsi_oversold = 30
    context.atr_period = 10
    context.atr_mult = 2.0
    context.tp_perc = 0.015  # 止盈百分比
    context.sl_perc = 0.015   # 止损百分比
    context.use_both_indicators = True  # 同时使用RSI和SuperTrend
    
    # 设置月度重新选股
    run_monthly(rebalance, date_rule=1)
    
    # 初始化股票池和权重
    context.stock_pool = []
    
    # 新增止损参数 - 更宽松的数值
    context.fixed_stop_pct = 0.15      # 固定百分比止损（相对入场价）- 更宽松的15%
    context.trailing_stop_pct = 0.12     # 跟踪止损百分比 - 更宽松的12%
    context.atr_multiplier = 3.5         # ATR倍数止损 - 更宽松的3.5倍
    context.minimum_hold_days = 2        # 最少持有天数，避免过早止损
    context.max_stop_count = 2           # 单日最大止损股票数量，避免系统性踩踏
    
    # 这是啥
    context.vol_stop_threshold = 2.0     # 放量下跌止损阈值
    context.use_combined_stop_loss = True # 是否使用综合止损策略
    
    # 创建字典存储每个股票的跟踪最高价和持有天数
    context.trailing_highs = {}
    context.holding_days = {}
    context.stopped_today = 0            # 记录当日已止损股票数量
    context.last_stop_date = None        # 初始化最后止损日期

    # 修改风险股过滤相关配置
    context.filter_sentence = '过滤停牌;过滤涨停;过滤跌停;过滤PE为负;过滤ST;未停牌;'
    
    # 改进安全股票池获取逻辑
    log.info("初始化时获取安全股票池...")
    context.safe_stocks = []
    context.safe_stocks_cache = set()
    
    try:
        # 在init中调用get_iwencai，结果会保存在context.safe_stocks中
        get_iwencai(context.filter_sentence, 'safe_stocks')
        
        # 等待一小段时间让系统处理完成
        time.sleep(0.1)
        
        # 检查是否成功获取到安全股票池
        if hasattr(context, 'safe_stocks') and context.safe_stocks:
            context.safe_stocks_cache = set()
            for stock in context.safe_stocks:
                # 统一股票代码格式
                try:
                    normalized_stock = normalize_stock_code(stock)
                    context.safe_stocks_cache.add(normalized_stock)
                except:
                    # 如果无法标准化，保留原格式
                    context.safe_stocks_cache.add(stock)
            log.info("安全股票池初始化成功，共有{}只安全股票".format(len(context.safe_stocks_cache)))
        else:
            log.warn("安全股票池获取失败，将使用宽松的过滤策略")
            context.use_safe_filter = False
    except Exception as e:
        log.error("获取安全股票池时出错: {}，将使用宽松的过滤策略".format(e))
        context.use_safe_filter = False
    
    # 设置是否使用安全股票池过滤
    context.use_safe_filter = len(context.safe_stocks_cache) > 0

    # 读取消费环境预测数据
    try:
        context.environment_df = pd.read_csv('environment_predictions_new.csv', index_col=0)
        context.environment_df.index = pd.to_datetime(context.environment_df.index)
        log.info("成功读取环境预测数据，日期范围: %s 到 %s" % (context.environment_df.index.min(), context.environment_df.index.max()))
    except Exception as e:
        log.error("读取环境预测数据失败: %s" % e)
        # 如果读取失败，设置默认环境
        context.environment_df = None
        context.environment = "良好"
    
    # 读取消费beta数据
    try:
        context.consumer_betas = pd.read_csv('consumer_betas.csv', index_col=0, dtype=str)
        # 将索引（日期）转换为datetime格式
        context.consumer_betas.index = pd.to_datetime(context.consumer_betas.index)
        # 确保列名（股票代码）是字符串格式
        context.consumer_betas.columns = context.consumer_betas.columns.astype(str)
        log.info("成功读取beta数据，日期范围: %s 到 %s" % (context.consumer_betas.index.min(), context.consumer_betas.index.max()))
        log.info("股票数量: %s" % len(context.consumer_betas.columns))
    except Exception as e:
        log.error("读取consumer_betas.csv时出错: %s" % e)
    
    # 读取赢家分类数据
    try:
        context.classification_df = pd.read_csv('classification_results.csv', index_col=0, dtype=str)
        context.classification_df.index = pd.to_datetime(context.classification_df.index)
        context.classification_df.columns = context.classification_df.columns.astype(str)
        log.info("成功读取分类数据，日期范围: %s 到 %s" % (context.classification_df.index.min(), context.classification_df.index.max()))
        log.info("股票数量: %s" % len(context.classification_df.columns))
    except Exception as e:
        log.error("读取classification_results.csv时出错: %s" % e)
        
     # 标记是否需要进行初始选股
    context.need_initial_rebalance = True

def is_valid_stock_code(stock_code):
    """检查股票代码是否有效"""
    # 检查是否有正确的后缀
    if '.' not in stock_code:
        return False
    
    # 分割代码和市场
    code, market = stock_code.split('.')
    
    # 检查市场是否有效
    if market not in ['SH', 'SZ']:
        return False
    
    # 检查代码是否为数字且长度合适
    if not code.isdigit():
        return False
    
    # 上海市场代码以6、9开头且长度为6
    if market == 'SH' and (not code.startswith(('6', '9', '5', '7', '0', '1', '2', '3', '4', '8')) or len(code) != 6):
        return False
    
    # 深圳市场代码以0、2、3开头且长度为6
    if market == 'SZ' and (not code.startswith(('0', '2', '3')) or len(code) != 6):
        return False
    
    return True

def normalize_stock_code(stock_code):
    """标准化股票代码格式"""
    # 如果有点，则去掉点和后面的部分
    if '.' in stock_code:
        stock_code = stock_code.split('.')[0]
    
    # 确保股票代码是数字，并且补足到6位
    if stock_code.isdigit():
        stock_code = stock_code.zfill(6)  # 补足到6位，前面加0
    else:
        raise ValueError("股票代码必须是纯数字或以数字开头的有效格式")
    
    # 转换为字符串并返回
    stock_code = str(stock_code)
    stock_code = normalize_symbol(stock_code,'stock')
    return stock_code

def get_current_environment(context, current_date):
    """获取当前消费环境预测"""
    if context.environment_df is None:
        return "良好"  # 默认环境
    
    # 找到最接近当前日期的环境预测
    closest_date = min(context.environment_df.index, key=lambda x: abs(x - current_date))
    
    # 如果日期差距过大，返回默认值
    if abs((closest_date - current_date).days) > 45:
        log.warn("环境预测数据时间差太大，使用默认环境")
        return "良好"
    
    # 获取环境预测
    environment = context.environment_df.loc[closest_date, 'environment']
    log.info("当前消费环境预测: %s (基于%s的数据)" % (environment, closest_date.strftime('%Y-%m-%d')))
    return environment

def calculate_atr(high, low, close, period=14):
    """手动计算ATR指标"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr2[0] = 0  # 首个元素没有前一日数据
    tr3 = np.abs(low - np.roll(close, 1))
    tr3[0] = 0  # 首个元素没有前一日数据
    
    # 计算TR (True Range)
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    
    # 计算ATR
    atr = np.zeros_like(close)
    atr[:period] = np.NaN
    atr[period] = np.mean(tr[:period+1])
    
    for i in range(period+1, len(close)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
    return atr

def calculate_supertrend(high, low, close, period=10, multiplier=2.0):
    """计算SuperTrend指标"""
    # 计算ATR
    atr = calculate_atr(high, low, close, period)
    # print("ATR:", atr)

    # 计算基础带宽
    hl2 = (high + low) / 2
    # print("HL2:", hl2)

    # 初始化上轨和下轨
    upperband = np.zeros_like(close)
    lowerband = np.zeros_like(close)

    # 初始化SuperTrend和方向
    supertrend = np.zeros_like(close)
    direction = np.zeros_like(close)

    # 填充前period个元素为NaN
    supertrend[:period] = np.NaN
    direction[:period] = np.NaN

    # 计算初始上下轨
    for i in range(period, len(close)):
        upperband[i] = hl2[i] + (multiplier * atr[i])
        lowerband[i] = hl2[i] - (multiplier * atr[i])

    # 初始设置第一个有效值的方向和SuperTrend
    if period < len(close):
        # 初始方向设置
        direction[period] = 1 if close[period] > (hl2[period] + multiplier * atr[period]) else -1
        # 根据方向设置初始SuperTrend值
        if direction[period] == 1:
            supertrend[period] = lowerband[period]
        else:
            supertrend[period] = upperband[period]

    # 计算剩余的SuperTrend值
    for i in range(period+1, len(close)):
        # 更新根据前一个趋势调整上下轨
        if supertrend[i-1] == upperband[i-1]:
            # 前一个趋势是下跌
            if close[i] > upperband[i-1] * 0.855:  # 放宽条件
                direction[i] = 1
                supertrend[i] = lowerband[i]
            else:
                direction[i] = -1
                supertrend[i] = upperband[i]
        else:
            # 前一个趋势是上涨
            if close[i] < lowerband[i-1] * 1.01:  # 放宽条件
                direction[i] = -1
                supertrend[i] = upperband[i]
            else:
                direction[i] = 1
                supertrend[i] = lowerband[i]

    #     # 打印调试信息
    #     print("Day:" ,i, "Close", close[i], "Upperband", upperband[i-1], "Lowerband",  lowerband[i-1], "Direction", direction[i])

    # # 输出最终结果
    # print("SuperTrend计算完成，最后5个值：", supertrend[-5:])
    # print("方向最后5个值：", direction[-5:])
    # print("收盘价最后5个值：", close[-5:])

    return supertrend, direction

def check_stock_is_st(stock_code, bar_dict):
    """检查股票是否为ST股票"""
    try:
        # 使用get_security_info检查股票信息，更可靠
        security_info = get_security_info(stock_code)
        if security_info is not None:
            # 检查股票名称是否包含"ST"
            if security_info.display_name and ('ST' in security_info.display_name or '*' in security_info.display_name):
                return True
    except Exception as e:
        log.warn("获取股票{}的ST信息失败: {}".format(stock_code, e))
    
    # 如果无法通过名称判断，尝试通过bar_dict检查is_st属性
    try:
        if bar_dict and stock_code in bar_dict:
            return bar_dict[stock_code].is_st
    except Exception:
        pass
    
    return False

def update_safe_stocks_cache(context, bar_dict):
    """
    更新安全股票池缓存，使用更安全的方式判断股票是否安全
    这个函数不使用get_iwencai，可以在任何地方调用
    """
    try:
        # 如果没有启用安全股票池过滤，则跳过更新
        if not hasattr(context, 'use_safe_filter') or not context.use_safe_filter:
            log.info("未启用安全股票池过滤，跳过更新")
            return
        
        # 确保安全股票池缓存存在
        if not hasattr(context, 'safe_stocks_cache'):
            context.safe_stocks_cache = set()
        
        # 如果安全股票池缓存为空，尝试重新初始化
        if not context.safe_stocks_cache and hasattr(context, 'safe_stocks') and context.safe_stocks:
            log.info("重新初始化安全股票池缓存")
            for stock in context.safe_stocks:
                # 统一股票代码格式
                try:
                    normalized_stock = normalize_stock_code(stock)
                    context.safe_stocks_cache.add(normalized_stock)
                except:
                    # 如果无法标准化，保留原格式
                    context.safe_stocks_cache.add(stock)
        
        # 动态过滤ST股票（从当前缓存中移除ST股票）
        if bar_dict and context.safe_stocks_cache:
            stocks_to_remove = []
            for stock_code in list(context.safe_stocks_cache):  # 使用list()避免在迭代中修改set
                # 检查是否是ST股
                if check_stock_is_st(stock_code, bar_dict):
                    stocks_to_remove.append(stock_code)
                    log.info("股票 {} 是ST股，从安全股票池中移除".format(stock_code))
            
            # 移除ST股票
            for stock_code in stocks_to_remove:
                context.safe_stocks_cache.discard(stock_code)  # 使用discard而不是remove，避免KeyError
        
        log.info("更新安全股票池缓存完成，共有{}只安全股票".format(len(context.safe_stocks_cache)))
        
        # 如果安全股票池为空，禁用安全过滤
        if not context.safe_stocks_cache:
            log.warn("安全股票池为空，禁用安全股票过滤")
            context.use_safe_filter = False
            
    except Exception as e:
        log.error("更新安全股票池缓存出错: {}".format(e))
        # 出错时禁用安全过滤，确保策略能继续运行
        context.use_safe_filter = False

def is_safe_stock(context, code):
    """判断股票是否安全，支持多种股票代码格式匹配"""
    if not hasattr(context, 'safe_stocks_cache') or not context.safe_stocks_cache:
        return True  # 如果没有安全股票池，默认认为是安全的
    
    # 直接匹配
    if code in context.safe_stocks_cache:
        return True
    
    # 尝试多种格式匹配
    try:
        # 提取纯数字代码
        if '.' in code:
            base_code = code.split('.')[0]
        else:
            base_code = code
            
        # 生成可能的格式
        possible_codes = set()
        possible_codes.add(base_code)  # 纯数字格式，如'000001'
        possible_codes.add(base_code + '.SZ')  # 深圳格式
        possible_codes.add(base_code + '.SH')  # 上海格式
        
        # 如果原始代码包含市场后缀，也添加原始格式
        if '.' in code:
            possible_codes.add(code)
        else:
            # 根据股票代码判断可能的市场
            if base_code.startswith(('6', '9', '5')):
                possible_codes.add(base_code + '.SH')
            elif base_code.startswith(('0', '2', '3')):
                possible_codes.add(base_code + '.SZ')
        
        # 检查是否有任何一种格式在安全股票池中
        for possible_code in possible_codes:
            if possible_code in context.safe_stocks_cache:
                return True
                
    except Exception as e:
        log.warn("检查股票{}安全性时出错: {}".format(code, e))
    
    return False

# 月度重新选股函数
def rebalance(context, bar_dict):
    # 更新安全股票池缓存，而不是使用get_iwencai
    update_safe_stocks_cache(context, bar_dict)
    
    current_date = get_datetime()
    log.info("开始月度调仓，当前日期: %s" % current_date)
    
    # 打印安全股票池状态信息
    if hasattr(context, 'use_safe_filter') and context.use_safe_filter:
        log.info("安全股票池过滤: 启用，共有{}只安全股票".format(len(context.safe_stocks_cache)))
        if len(context.safe_stocks_cache) > 0:
            # 显示前10只安全股票作为示例
            sample_stocks = list(context.safe_stocks_cache)[:10]
            log.info("安全股票池示例: {}".format(sample_stocks))
    else:
        log.info("安全股票池过滤: 未启用，将使用宽松过滤策略")
    
    # 获取当前消费环境
    context.environment = get_current_environment(context, current_date)
    
    # 根据消费环境选择策略
    if context.environment == "良好":
        log.info("当前消费环境: 良好，选择高beta股票")
        # 找到最接近当前日期的beta数据
        beta_dates = context.consumer_betas.index
        closest_beta_date = min(beta_dates, key=lambda x: abs(x - current_date))
        
        if abs((closest_beta_date - current_date).days) > 45:  # 如果时间差太大，使用默认策略
            log.warn("Beta数据时间差太大，使用默认股票池")
            return
        
        log.info("使用最接近的beta日期: %s" % closest_beta_date.strftime('%Y-%m-%d'))
        
        # 获取当月的beta值
        current_betas = context.consumer_betas.loc[closest_beta_date]
        
        # 移除缺失值
        current_betas = current_betas.dropna()
        
        if len(current_betas) == 0:
            log.warn("当月没有有效的beta数据，使用默认股票池")
            return
        
        # 选择beta最高的30支股票（良好环境下选择高beta）
        max_stocks = 30
        selected_stocks = current_betas.sort_values(ascending=False).head(max_stocks)
        first_pool = list(selected_stocks.index)
        
        log.info("第一次筛选 - 选出 %d 只高beta股票" % len(first_pool))
        
        # 第二次筛选 - 选择赢家
        # 找到最接近当前日期的分类结果
        class_dates = context.classification_df.index
        closest_class_date = min(class_dates, key=lambda x: abs(x - current_date))
        
        if abs((closest_class_date - current_date).days) > 45:  # 如果时间差太大，使用第一次筛选结果
            log.warn("分类数据时间差太大，使用第一次筛选结果")
            winners = first_pool
            winner_betas = [selected_stocks[stock] for stock in winners]
        else:
            log.info("使用最接近的分类日期: %s" % closest_class_date.strftime('%Y-%m-%d'))
            
            # 获取当月的分类结果
            current_classification = context.classification_df.loc[closest_class_date]
            
            # 从第一次筛选的结果中选择赢家
            winners = []
            winner_betas = []
            
            for stock in first_pool:
                if stock in current_classification and current_classification[stock] == 1:  # 赢家
                    winners.append(stock)
                    winner_betas.append(selected_stocks[stock])
            
            log.info("第二次筛选（选择赢家）- 选出 %d 只股票" % len(winners))
        
        if len(winners) == 0:
            log.warn("没有找到赢家股票，使用第一次筛选结果")
            winners = first_pool
            winner_betas = [selected_stocks[stock] for stock in winners]
        
        # 验证股票代码
        valid_winners = []
        valid_betas = []
        
        for i, stock in enumerate(winners):
            # 标准化股票代码
            normalized_code = normalize_stock_code(stock)
            if normalized_code and is_valid_stock_code(normalized_code):
                valid_winners.append(normalized_code)
                valid_betas.append(winner_betas[i])
            else:
                log.warn("股票代码 %s 格式无效，已跳过" % stock)
        
        log.info("验证股票代码后剩余 %d 只有效股票" % len(valid_winners))
        
        if len(valid_winners) == 0:
            log.error("没有有效的股票代码，跳过本次调仓")
            return
        
        
        
        
        
        
        # 改进的安全股票过滤逻辑，添加回退机制（良好环境）
        final_winners = []
        final_betas = []
        filtered_stocks = []
        
        # 如果启用了安全股票池过滤
        if context.use_safe_filter:
            for i, stock_code in enumerate(valid_winners):
                # 检查股票是否安全
                if is_safe_stock(context, stock_code):
                    final_winners.append(stock_code)
                    final_betas.append(valid_betas[i])
                else:
                    filtered_stocks.append(stock_code)
                    log.warn("股票 {} 不在安全股票池中，已过滤".format(stock_code))
            
            log.info("安全股票池过滤: 保留 {} 只股票, 过滤 {} 只股票".format(len(final_winners), len(filtered_stocks)))
            
            # 回退机制：如果过滤后股票太少，放宽过滤条件
            if len(final_winners) < 5:
                log.warn("安全过滤后股票数量过少({})，使用宽松过滤策略".format(len(final_winners)))
                # 只过滤明显的风险股票（如ST股票）
                final_winners = []
                final_betas = []
                for i, stock_code in enumerate(valid_winners):
                    # 只检查ST股票，不检查安全股票池
                    if not check_stock_is_st(stock_code, None):
                        final_winners.append(stock_code)
                        final_betas.append(valid_betas[i])
                    else:
                        log.warn("股票 {} 是ST股，已过滤".format(stock_code))
                
                log.info("宽松过滤后剩余 {} 只股票".format(len(final_winners)))
        else:
            # 如果不使用安全股票池过滤，只过滤ST股票
            log.info("未启用安全股票池过滤，仅过滤ST股票")
            for i, stock_code in enumerate(valid_winners):
                if not check_stock_is_st(stock_code, None):
                    final_winners.append(stock_code)
                    final_betas.append(valid_betas[i])
                else:
                    log.warn("股票 {} 是ST股，已过滤".format(stock_code))
            
            log.info("ST股票过滤后剩余 {} 只股票".format(len(final_winners)))
        
        # 最终检查：如果还是没有股票，使用原始列表
        if len(final_winners) == 0:
            log.error("所有过滤后都没有剩余股票，使用原始股票列表（无过滤）")
            final_winners = valid_winners
            final_betas = valid_betas
            
        log.info("最终确定 {} 只股票进入投资组合（良好环境）".format(len(final_winners)))
        
        # 使用最终确定的股票列表继续
        valid_winners = final_winners
        valid_betas = final_betas
        
        # 根据beta值排序计算权重
        sorted_indices = np.argsort(valid_betas)[::-1]  # 高beta优先（良好环境）
        
        # 计算权重
        total_stocks = len(valid_winners)
        weights = []
        for i in range(total_stocks):
            weight = (total_stocks - i) / (total_stocks * (total_stocks + 1) / 2)
            weights.append(weight)
        
        # 创建最终股票池
        context.stock_pool = []
        for idx, sort_idx in enumerate(sorted_indices):
            stock_code = valid_winners[sort_idx]
            context.stock_pool.append({
                'code': stock_code,
                'name': 'Stock_%s' % stock_code,
                'weight': weights[idx],
                'rank': idx + 1
            })
        
        log.info("最终股票池 - 共 %d 只股票" % len(context.stock_pool))
        
        # 进行实际调仓
        # 先清仓所有持仓
        for position in context.portfolio.positions:
            if position not in [stock['code'] for stock in context.stock_pool]:
                order_target_percent(position, 0)
                log.info("清仓非股票池股票: %s" % position)
        
        # 再调整目标持仓
        for stock in context.stock_pool:
            stock_code = stock['code']
            target_percent = stock['weight']
            # 使用标准格式的股票代码下单
            log.info("调仓 %s 目标权重: %.2f%%" % (stock_code, target_percent*100))
            try:
                # 验证一下该股票是否可交易
                sec_info = get_security_info(stock_code)
                if sec_info is None:
                    log.error("获取股票 %s 信息失败，跳过调仓" % stock_code)
                    continue
                order_target_percent(stock_code, target_percent)
            except Exception as e:
                log.error("调仓 %s 失败: %s" % (stock_code, e))
    else:
        log.info("当前消费环境: 疲软，选择低beta股票")
        # 找到最接近当前日期的beta数据
        beta_dates = context.consumer_betas.index
        closest_beta_date = min(beta_dates, key=lambda x: abs(x - current_date))
        
        if abs((closest_beta_date - current_date).days) > 45:
            log.warn("Beta数据时间差太大，使用默认股票池")
            return
        
        log.info("使用最接近的beta日期: %s" % closest_beta_date.strftime('%Y-%m-%d'))
        
        # 获取当月的beta值
        current_betas = context.consumer_betas.loc[closest_beta_date]
        
        # 移除缺失值
        current_betas = current_betas.dropna()
        
        if len(current_betas) == 0:
            log.warn("当月没有有效的beta数据，使用默认股票池")
            return
        
        # 选择beta最高的30支股票（良好环境下选择高beta）
        max_stocks = 30
        selected_stocks = current_betas.sort_values(ascending=True).head(max_stocks)
        first_pool = list(selected_stocks.index)
        
        log.info("第一次筛选 - 选出 %d 只高beta股票" % len(first_pool))
        
        # 第二次筛选 - 选择赢家
        # 找到最接近当前日期的分类结果
        class_dates = context.classification_df.index
        closest_class_date = min(class_dates, key=lambda x: abs(x - current_date))
        
        if abs((closest_class_date - current_date).days) > 45:
            log.warn("分类数据时间差太大，使用第一次筛选结果")
            winners = first_pool
            winner_betas = [selected_stocks[stock] for stock in winners]
        else:
            log.info("使用最接近的分类日期: %s" % closest_class_date.strftime('%Y-%m-%d'))
            
            # 获取当月的分类结果
            current_classification = context.classification_df.loc[closest_class_date]
            
            # 从第一次筛选的结果中选择赢家
            winners = []
            winner_betas = []
            
            for stock in first_pool:
                if stock in current_classification and current_classification[stock] == 1:  # 赢家
                    winners.append(stock)
                    winner_betas.append(selected_stocks[stock])
            
            log.info("第二次筛选（选择赢家）- 选出 %d 只股票" % len(winners))
        
        if len(winners) == 0:
            log.warn("没有找到赢家股票，使用第一次筛选结果")
            winners = first_pool
            winner_betas = [selected_stocks[stock] for stock in winners]
        
        # 验证股票代码
        valid_winners = []
        valid_betas = []
        
        for i, stock in enumerate(winners):
            # 标准化股票代码
            normalized_code = normalize_stock_code(stock)
            if normalized_code and is_valid_stock_code(normalized_code):
                valid_winners.append(normalized_code)
                valid_betas.append(winner_betas[i])
            else:
                log.warn("股票代码 %s 格式无效，已跳过" % stock)
        
        log.info("验证股票代码后剩余 %d 只有效股票" % len(valid_winners))
        
        if len(valid_winners) == 0:
            log.error("没有有效的股票代码，跳过本次调仓")
            return
        
        
        
        # 改进的安全股票过滤逻辑，添加回退机制（疲软环境）
        final_winners = []
        final_betas = []
        filtered_stocks = []
        
        # 如果启用了安全股票池过滤
        if context.use_safe_filter:
            for i, stock_code in enumerate(valid_winners):
                # 检查股票是否安全
                if is_safe_stock(context, stock_code):
                    final_winners.append(stock_code)
                    final_betas.append(valid_betas[i])
                else:
                    filtered_stocks.append(stock_code)
                    log.warn("股票 {} 不在安全股票池中，已过滤".format(stock_code))
            
            log.info("安全股票池过滤: 保留 {} 只股票, 过滤 {} 只股票".format(len(final_winners), len(filtered_stocks)))
            
            # 回退机制：如果过滤后股票太少，放宽过滤条件
            if len(final_winners) < 5:
                log.warn("安全过滤后股票数量过少({})，使用宽松过滤策略".format(len(final_winners)))
                # 只过滤明显的风险股票（如ST股票）
                final_winners = []
                final_betas = []
                for i, stock_code in enumerate(valid_winners):
                    # 只检查ST股票，不检查安全股票池
                    if not check_stock_is_st(stock_code, None):
                        final_winners.append(stock_code)
                        final_betas.append(valid_betas[i])
                    else:
                        log.warn("股票 {} 是ST股，已过滤".format(stock_code))
                
                log.info("宽松过滤后剩余 {} 只股票".format(len(final_winners)))
        else:
            # 如果不使用安全股票池过滤，只过滤ST股票
            log.info("未启用安全股票池过滤，仅过滤ST股票")
            for i, stock_code in enumerate(valid_winners):
                if not check_stock_is_st(stock_code, None):
                    final_winners.append(stock_code)
                    final_betas.append(valid_betas[i])
                else:
                    log.warn("股票 {} 是ST股，已过滤".format(stock_code))
            
            log.info("ST股票过滤后剩余 {} 只股票".format(len(final_winners)))
        
        # 最终检查：如果还是没有股票，使用原始列表
        if len(final_winners) == 0:
            log.error("所有过滤后都没有剩余股票，使用原始股票列表（无过滤）")
            final_winners = valid_winners
            final_betas = valid_betas
            
        log.info("最终确定 {} 只股票进入投资组合（疲软环境）".format(len(final_winners)))
        
        # 使用最终确定的股票列表继续
        valid_winners = final_winners
        valid_betas = final_betas
        
        # 根据beta值排序计算权重
        sorted_indices = np.argsort(valid_betas)  # 低beta优先（疲软环境）
        
        # 计算权重
        total_stocks = len(valid_winners)
        weights = []
        for i in range(total_stocks):
            weight = (total_stocks - i) / (total_stocks * (total_stocks + 1) / 2)
            weights.append(weight)
        
        # 创建最终股票池
        context.stock_pool = []
        for idx, sort_idx in enumerate(sorted_indices):
            stock_code = valid_winners[sort_idx]
            context.stock_pool.append({
                'code': stock_code,
                'name': 'Stock_%s' % stock_code,
                'weight': weights[idx],
                'rank': idx + 1
            })
        
        log.info("最终股票池 - 共 %d 只股票" % len(context.stock_pool))
        
        # 进行实际调仓
        # 先清仓所有持仓
        for position in context.portfolio.positions:
            if position not in [stock['code'] for stock in context.stock_pool]:
                order_target_percent(position, 0)
                log.info("清仓非股票池股票: %s" % position)
        
        # 再调整目标持仓
        for stock in context.stock_pool:
            stock_code = stock['code']
            target_percent = stock['weight']
            # 使用标准格式的股票代码下单
            log.info("调仓 %s 目标权重: %.2f%%" % (stock_code, target_percent*100))
            try:
                # 验证一下该股票是否可交易
                sec_info = get_security_info(stock_code)
                if sec_info is None:
                    log.error("获取股票 %s 信息失败，跳过调仓" % stock_code)
                    continue
                order_target_percent(stock_code, target_percent)
            except Exception as e:
                log.error("调仓 %s 失败: %s" % (stock_code, e))

def check_stop_loss(context, code, position, close_prices, high_prices, low_prices, current_date):
    """更平衡的止损策略，减少过早退出"""
    # 获取当前价格
    current_price = close_prices[-1]
    stop_triggered = False
    stop_reason = ""
    
    # 重置当日止损计数（如果是新的交易日）
    if not hasattr(context, 'last_stop_date') or context.last_stop_date is None or context.last_stop_date != current_date.date():
        context.last_stop_date = current_date.date()
        context.stopped_today = 0
    
    # 检查是否达到当日最大止损数量
    if context.stopped_today >= context.max_stop_count:
        return False
    
    # 确保持有天数字典存在
    if not hasattr(context, 'holding_days'):
        context.holding_days = {}
        
    # 更新持有天数计数
    if code not in context.holding_days:
        context.holding_days[code] = 1
    else:
        context.holding_days[code] += 1
    
    # 如果没超过最小持有天数，不触发止损
    if context.holding_days[code] < context.minimum_hold_days:
        return False
    
    # 确保跟踪最高价字典存在
    if not hasattr(context, 'trailing_highs'):
        context.trailing_highs = {}
        
    # 初始化或更新该股票的跟踪最高价
    if code not in context.trailing_highs:
        context.trailing_highs[code] = max(position.avg_price, current_price)
    elif current_price > context.trailing_highs[code]:
        context.trailing_highs[code] = current_price
    
    # 计算亏损百分比（相对入场价）
    loss_pct = (current_price / position.avg_price) - 1
    
    # 策略1: 只有大幅亏损才触发固定止损（15%）
    if loss_pct < -context.fixed_stop_pct:
        stop_triggered = True
        stop_reason = "大幅亏损止损：亏损 %.2f%%" % (loss_pct*100)
    
    # 策略2: 对有盈利的头寸使用跟踪止损（从高点回撤12%）
    if not stop_triggered and loss_pct > 0.05:  # 只对盈利5%以上的头寸使用跟踪止损
        trailing_high = context.trailing_highs[code]
        drop_pct = (trailing_high - current_price) / trailing_high
        
        if drop_pct > context.trailing_stop_pct:
            stop_triggered = True
            stop_reason = "盈利保护止损：从最高点 %.2f 下跌 %.2f%%" % (trailing_high, drop_pct*100)
    
    # 策略3: 只在明确的下跌趋势中使用ATR止损
    if not stop_triggered:
        # 计算短期趋势（如5日均线VS10日均线）
        if len(close_prices) >= 10:
            ma5 = np.mean(close_prices[-5:])
            ma10 = np.mean(close_prices[-10:])
            is_downtrend = ma5 < ma10
            
            if is_downtrend and loss_pct < 0:  # 只在下跌趋势+亏损时考虑ATR止损
                atr_period = min(context.atr_period, len(close_prices) - 1)
                if atr_period >= 3:
                    atr = calculate_atr(high_prices, low_prices, close_prices, period=atr_period)
                    current_atr = atr[-1]
                    
                    # 使用更宽松的ATR倍数
                    stop_price = position.avg_price - (context.atr_multiplier * current_atr)
                    
                    if current_price < stop_price:
                        stop_triggered = True
                        stop_reason = "趋势性ATR止损：价格 %.2f 低于止损价 %.2f" % (current_price, stop_price)
    
    # 记录止损原因并更新计数
    if stop_triggered:
        log.info("股票 %s 触发止损：%s, 当前价格=%.2f, 买入价=%.2f, 持有天数=%d" % 
                (code, stop_reason, current_price, position.avg_price, context.holding_days[code]))
        context.stopped_today += 1
        # 重置该股票的持有记录
        if code in context.holding_days:
            del context.holding_days[code]
        if code in context.trailing_highs:
            del context.trailing_highs[code]
    
    return stop_triggered

def before_trading(context):
    log.info("开始交易日准备")
    
    # 检查当前持仓
    for code in context.portfolio.positions:
        position = context.portfolio.positions[code]
        if position.amount > 0:
            log.info("当前持仓 %s: %d 股, 市值: %.2f, 成本: %.2f" % 
                    (code, position.amount, position.market_value, position.cost_basis))
            
            # 使用缓存的安全股票池检查持仓股票是否安全
            if code not in context.safe_stocks_cache:
                log.warn("持仓股票 {} 不在安全股票池中，考虑在今日交易中卖出".format(code))

def handle_bar(context, bar_dict):
    # 每个交易日开始时更新安全股票池缓存
    update_safe_stocks_cache(context, bar_dict)
    
    # 如果需要进行初始选股
    if context.need_initial_rebalance:
        log.info("策略首次运行，执行初始选股...")
        rebalance(context, bar_dict)
        context.need_initial_rebalance = False  # 标记初始选股已完成
        return  # 第一次执行完初始选股后直接返回，不执行其他交易逻辑
    
    # 遍历当前持仓的每只股票
    current_date = get_datetime()
    
    # 获取所有持仓股票的代码列表
    position_codes = [code for code in context.portfolio.positions if context.portfolio.positions[code].amount > 0]
    
    if not position_codes:
        log.info("当前没有持仓股票，跳过RSI计算")
        return
    
    log.info("开始处理 %d 只持仓股票" % len(position_codes))
    
    # 首先尝试批量获取历史数据，这可能在SuperMind中更有效
    try:
        # 直接获取所有股票的历史数据
        all_hist_data = history(position_codes, ['close', 'high', 'low'], 15, '1d', False)
        log.info("批量获取历史数据成功，返回数据类型: %s" % type(all_hist_data).__name__)
    except Exception as e:
        log.error("批量获取历史数据失败: %s，将尝试逐只股票获取" % e)
        all_hist_data = None
    
    # 预先为所有股票获取历史价格数据
    hist_data_dict = {}
    
    # 如果批量获取成功，直接使用批量数据
    if all_hist_data is not None:
        for code in position_codes:
            if code in all_hist_data and len(all_hist_data[code]) >= 5:
                hist_data_dict[code] = all_hist_data[code]
                # log.info("股票 %s 成功获取到 %d 条历史数据" % (code, len(all_hist_data[code])))
            else:
                log.warn("批量获取中股票 %s 数据不足" % code)
    else:
        # 如果批量获取失败，逐个获取
        for code in position_codes:
            try:
                # 使用更短的时间窗口获取历史数据
                hist_data = history(code, ['close', 'high', 'low'], 15, '1d', False)
                
                # 添加详细的调试信息来了解hist_data的结构
                log.info("股票 %s 历史数据获取结果类型: %s" % (code, type(hist_data).__name__))
                
                if hist_data is not None:
                    if isinstance(hist_data, dict) and code in hist_data:
                        log.info("股票 %s 有效数据条数: %d" % (code, len(hist_data[code])))
                        if len(hist_data[code]) >= 5:
                            hist_data_dict[code] = hist_data[code]
                        else:
                            log.warn("股票 %s 历史数据条数不足: %d < 5" % (code, len(hist_data[code])))
                    else:
                        # 可能SuperMind的返回结构与我们预期不同
                        log.info("尝试直接使用hist_data，数据类型: %s" % type(hist_data).__name__)
                        if hasattr(hist_data, '__len__') and len(hist_data) >= 5:
                            hist_data_dict[code] = hist_data
                            log.info("直接使用hist_data成功获取 %d 条数据" % len(hist_data))
                        else:
                            log.warn("股票 %s 无法解析历史数据" % code)
                else:
                    log.warn("股票 %s 获取的历史数据为None" % code)
            except Exception as e:
                log.error("获取股票 %s 历史数据时出错: %s" % (code, e))
    
    # 筛选出有历史数据的股票代码
    valid_codes = list(hist_data_dict.keys())
    log.info("共有 %d 只股票有足够的历史数据可以处理" % len(valid_codes))
    
    if not valid_codes:
        log.warn("没有任何股票有足够的历史数据，跳过交易信号判断")
        return
    
    # 尝试手动计算RSI
    rsi_values_dict = {}
    # 遍历当前持仓的每只股票
    current_date = get_datetime()
    start_date = (current_date - timedelta(days=30)).strftime('%Y%m%d')
    end_date = current_date.strftime('%Y%m%d')
    
    # 使用get_sfactor_data获取RSI指标数据
    rsi_values_dict = get_sfactor_data(start_date, end_date, position_codes, ['rsi'])
    
    if rsi_values_dict is None or len(rsi_values_dict) == 0:
        log.warn("未获取到RSI数据，跳过交易信号判断")
        return
    # print(rsi_values_dict)
    # print(rsi_values_dict['rsi'])
    
    # for code in valid_codes:
    #     try:
    #         # 获取close价格数据，适应不同的数据结构
    #         data = hist_data_dict[code]
    #         if isinstance(data, dict) and 'close' in data:
    #             close_prices = data['close'].values if hasattr(data['close'], 'values') else data['close']
    #         elif hasattr(data, 'close'):
    #             close_prices = data.close.values if hasattr(data.close, 'values') else data.close
    #         else:
    #             # 如果无法获取close，尝试其他方式
    #             log.warn("无法从历史数据中获取close价格，尝试其他方式")
    #             continue
            
    #         # 打印close_prices确认数据正确
    #         log.info("股票 %s close_prices类型: %s, 长度: %d" % 
    #                 (code, type(close_prices).__name__, len(close_prices)))
            
    #         # 手动计算RSI
    #         rsi_values = calculate_rsi(close_prices, period=min(len(close_prices)-1, context.rsi_period))
    #         rsi_values_dict[code] = rsi_values
    #         log.info("股票 %s 手动计算RSI成功，当前RSI: %.2f" % (code, rsi_values[-1]))
    #     except Exception as e:
    #         log.error("计算股票 %s 的RSI时出错: %s" % (code, e))
    
    # 对每个持仓股票进行交易信号判断
    for code in valid_codes:
        if code not in rsi_values_dict['rsi'].index:
            log.warn("股票 %s 无法计算RSI，跳过处理" % code)
            continue
            
        position = context.portfolio.positions[code]
        
        try:
            # 获取价格数据，适应不同的数据结构
            data = hist_data_dict[code]
            if isinstance(data, dict):
                if 'close' in data:
                    close_prices = data['close'].values if hasattr(data['close'], 'values') else data['close']
                    high_prices = data['high'].values if hasattr(data['high'], 'values') else data['high']
                    low_prices = data['low'].values if hasattr(data['low'], 'values') else data['low']
                else:
                    log.warn("数据字典中没有close/high/low字段")
                    continue
            elif hasattr(data, 'close') and hasattr(data, 'high') and hasattr(data, 'low'):
                close_prices = data.close.values if hasattr(data.close, 'values') else data.close
                high_prices = data.high.values if hasattr(data.high, 'values') else data.high
                low_prices = data.low.values if hasattr(data.low, 'values') else data.low
            else:
                log.warn("无法从历史数据中获取价格信息")
                continue
            
            # 执行更温和的止损检查
            if context.use_combined_stop_loss and position.amount > 0:
                if check_stop_loss(context, code, position, close_prices, high_prices, low_prices, current_date):
                    # 使用分批减仓策略，不是一次性清仓
                    current_weight = position.market_value / context.portfolio.portfolio_value
                    # 第一次触发止损只减持一半
                    new_target = current_weight * 0.5
                    order_target_percent(code, new_target)
                    log.info("止损减仓 %s 至 %.2f%% 仓位" % (code, new_target*100))
                    continue
            
            # 使用手动计算的RSI
            # rsi_values = rsi_values_dict['rsi'][code]
            # rsi_current = rsi_values[-1]
            # rsi_prev = rsi_values[-2] if len(rsi_values) > 1 else None
            
            # 提取对应股票的 RSI 值
            rsi_values = rsi_values_dict['rsi'].loc[code]
            # 获取当前值和前一个值
            rsi_current = rsi_values[-1]
            rsi_prev = rsi_values[-2] if len(rsi_values) > 1 else None
            
            
            
            
            
            
            # 计算SuperTrend指标 - 使用可用的数据点
            atr_period = min(context.atr_period, len(close_prices) - 1)
            if atr_period < 3:  # 太短的周期可能导致不可靠的结果
                log.warn("股票 %s 可用数据点过少，无法可靠计算SuperTrend" % code)
                continue
                
            supertrend, direction = calculate_supertrend(
                high_prices, 
                low_prices, 
                close_prices, 
                period=atr_period, 
                multiplier=context.atr_mult
            )
            
            # 判断当前趋势
            current_idx = len(close_prices) - 1
            is_uptrend = direction[current_idx] == 1
            is_downtrend = direction[current_idx] == -1
            
            # 计算RSI的超买超卖信号
            rsi_overbought = rsi_current > context.rsi_overbought
            rsi_oversold = rsi_current < context.rsi_oversold
            
            rsi_overbought_crossover = rsi_prev is not None and rsi_current > context.rsi_overbought and rsi_prev <= context.rsi_overbought
            rsi_oversold_crossover = rsi_prev is not None and rsi_current < context.rsi_oversold and rsi_prev >= context.rsi_oversold
            
            # log.info("股票 %s: RSI=%.2f, SuperTrend方向=%d, 收盘价=%.2f" % 
            #         (code, rsi_current, direction[current_idx], close_prices[current_idx]))
            
            # 交易信号处理
            if context.use_both_indicators:
                # 卖出信号: RSI超买且SuperTrend转为下跌趋势
                if (rsi_overbought or rsi_overbought_crossover) and is_downtrend:
                    log.info("股票 %s 产生卖出信号: RSI超买(%.2f) + SuperTrend下跌趋势" % (code, rsi_current))
                    order_target_percent(code, 0)  # 清仓
                
                # 买入信号: RSI超卖且SuperTrend转为上涨趋势
                elif (rsi_oversold or rsi_oversold_crossover) and is_uptrend:
                    # 买入前检查是否是安全股票
                    if not is_safe_stock(context, code):
                        log.warn("股票 {} 产生买入信号但不在安全股票池中，跳过买入".format(code))
                        continue
                    
                    # 查找该股票的目标权重
                    target_weight = 0
                    for stock in context.stock_pool:
                        if stock['code'] == code:
                            target_weight = stock['weight']
                            break
                    
                    current_weight = position.market_value / context.portfolio.portfolio_value
                    if current_weight < target_weight:
                        log.info("股票 %s 产生买入信号: RSI超卖(%.2f) + SuperTrend上涨趋势, 目标权重: %.2f%%" % 
                                (code, rsi_current, target_weight*100))
                        order_target_percent(code, target_weight)
                
                # 止损检查: 如果价格低于SuperTrend值的(1-sl_perc)，执行止损
                elif close_prices[current_idx] < supertrend[current_idx] * (1 - context.sl_perc) and is_downtrend:
                    log.info("股票 %s 触发止损: 当前价格=%.2f, SuperTrend=%.2f" % 
                            (code, close_prices[current_idx], supertrend[current_idx]))
                    order_target_percent(code, 0)  # 清仓
            else:
                # 只使用RSI
                if rsi_overbought_crossover:
                    log.info("股票 %s 产生卖出信号: RSI超买(%.2f)" % (code, rsi_current))
                    order_target_percent(code, 0)  # 清仓
                elif rsi_oversold_crossover:
                    # 买入前检查是否是安全股票
                    if not is_safe_stock(context, code):
                        log.warn("股票 {} 产生买入信号但不在安全股票池中，跳过买入".format(code))
                        continue
                    
                    # 查找该股票的目标权重
                    target_weight = 0
                    for stock in context.stock_pool:
                        if stock['code'] == code:
                            target_weight = stock['weight']
                            break
                    
                    current_weight = position.market_value / context.portfolio.portfolio_value
                    if current_weight < target_weight:
                        log.info("股票 %s 产生买入信号: RSI超卖(%.2f), 目标权重: %.2f%%" % 
                                (code, rsi_current, target_weight*100))
                        order_target_percent(code, target_weight)
        except Exception as e:
            log.error("处理股票 %s 时出错: %s" % (code, e))
        
def after_trading(context):
    log.info("交易日结束")
    # 计算当日收益
    log.info("当日收益率: %.2f%%" % (context.portfolio.returns*100))
    log.info("当前总资产: %.2f" % context.portfolio.portfolio_value)