# quant_for_contest2505

This project provides a streamlined solution for preprocessing data in a Tonghuashun SuperMind quant strategy. Due to SuperMind’s limited data (e.g., no consumption data) and weak computing resources, running the full strategy directly is impractical.

本仓库包含同花顺supermind策略的量化代码以及对应需要的数据文件。由于同花顺supermind并不能获取到很多数据（例如消费等）、并且supermind的服务器资源有限，又因为本策略需要处理消费数据和回报率等，直接在supermind上完全体运行可能有些不太现实。因此本策略采取了如下方案：先从CSMAR上手动下载消费数据和股票数据，然后再将预先处理完成的文件上传到supermind研究环境中以供策略运行。本项目为预先处理数据文件这个步骤提供一个方便的实现。以下将这称为【策略前置操作】。

## 步骤

> 如果你只是完成基础的实现【策略前置操作】，那么按照这四个步骤来就足够了。
>
> 一开始想整个一条龙的，但可惜没空累了，就这样吧，抱歉，路径什么的可以自行探索下

### 1. 数据准备（请不要抱怨，个人获取数据就是这样步骤）

> 需要注意的是，CSMAR经常改数据表标题和内容格式，实在折腾。不能保证时效性。
>
> 为了方便起见，在仓库中`data_sample`下有示例数据。如果你不想手动获取数据，那么请跳过这个第一个步骤，程序会自动从示例数据加载。

#### 1.1 消费数据获取

数据来自[国泰君安CSMAR](https://data.csmar.com/)，顶部tab选择【数据中心】，依次点击【单表查询】【经济研究系列】【宏观经济】【经济行业快讯】【社会消费品零售总额表(月)】全选所有字段，下载csv，解压得到`CME_Mretailsales.csv`。

<img src="./README.assets/Screenshot 2025-08-09 at 11.39.43 PM.png" alt="Screenshot 2025-08-09 at 11.39.43 PM" style="zoom:25%;" />

数据来自[国泰君安CSMAR](https://data.csmar.com/)，顶部tab选择【数据中心】，依次点击【单表查询】【经济研究系列】【宏观经济】【景气指数】【消费者景气指数月度文件】全选所有字段，下载csv，解压得到`CME_Mbcid2.csv`。

<img src="./README.assets/Screenshot 2025-08-09 at 11.19.21 PM.png" alt="Screenshot 2025-08-09 at 11.19.21 PM" style="zoom:25%;" />

数据来自[国泰君安CSMAR](https://data.csmar.com/)，顶部tab选择【数据中心】，依次点击【单表查询】【经济研究系列】【宏观经济】【价格指数】【居民消费价格分类指数月度文件】全选所有字段，下载csv，解压得到`CME_Mconsumerpriceratio.csv`。

<img src="./README.assets/Screenshot 2025-08-09 at 11.42.43 PM.png" alt="Screenshot 2025-08-09 at 11.42.43 PM" style="zoom:25%;" />

#### 1.2 股票数据获取

（当然这部分你可以用任意其他接口，需自行修改下数据标题或者`dataprocess.py`的读取）

> 注意⚠️：CSMAR这部分的数据存在单次5年的时间跨度的限制，如果获取的数据超过5年，那么获得的数据是多个zip，其内部的各自的文件名需要被重命名成`TRD_Dalyr1.csv`, `TRD_Dalyr2.csv`,`TRD_Dalyr3.csv`, 等。并放置在同一个文件夹中。
>
> 如果从CSMAR获取的数据跨度在5年之内，则不必理会上述这段的处理

数据来自[国泰君安CSMAR](https://data.csmar.com/)，顶部tab选择【数据中心】，依次点击【单表查询】【股票市场系列】【股票市场交易】【个股交易数据】【日个股回报率文件】，全选所有字段，下载csv，解压

<img src="./README.assets/Screenshot 2025-08-31 at 4.14.18 PM.png" alt="Screenshot 2025-08-31 at 4.14.18 PM" style="zoom:25%;" />

将获取到的csv数据文件放进`dataprocess`目录中。

### 2. 创建虚拟环境、依赖安装，运行各程序

- 使用conda

  ```python
  conda create -n env_name python==3.9
  conda activate env_name
  pip install requirements.txt
  python main.py
  ```

- 或者使用uv等pip venv工具

  ```python
  uv run main.py
  ```

> ~~主程序整合了零散的代码步骤，程序运行可能需要十多分钟（数据处理、消费环境处理、消费敏感性处理、股票输赢家处理），结束后程序会在`upload`目录中放置`consumer_betas.csv `,              `safe_stock.csv`, `classification_results.csv`, `environment_predictions_new.csv`这几个文件。~~

### 3. 上传

来到[同花顺supermind【我的研究】](https://quant.10jqka.com.cn/view/study-research.html)

<img src="./README.assets/Screenshot 2025-08-09 at 7.14.29 PM.png" alt="Screenshot 2025-08-09 at 7.14.29 PM" style="zoom:50%;" />

选择python3.8，启动服务器。稍等片刻之后进入jupyter-lab界面。点击【上传文件】按钮，上传`upload`文件夹内所有文件。完成后即可关闭这个页面。

<img src="./README.assets/Screenshot 2025-08-09 at 7.18.06 PM.png" alt="Screenshot 2025-08-09 at 7.18.06 PM" style="zoom: 25%;" />

<img src="./README.assets/Screenshot 2025-08-09 at 7.17.39 PM.png" alt="Screenshot 2025-08-09 at 7.15.43 PM" style="zoom: 25%;" />

### 4. 运行策略代码

来到[同花顺supermind【我的策略】](https://quant.10jqka.com.cn/view/study-index.html#/)，点击【新建策略】

<img src="./README.assets/Screenshot 2025-08-09 at 7.20.35 PM.png" alt="Screenshot 2025-08-09 at 7.20.35 PM" style="zoom: 25%;" />

将策略代码`supermind.py`内全部复制，粘贴进去。（CTRL+A全选）右侧可以调整自己想要的金额、时间等。

## 已知问题

- 回测区间不能超过5年，否则报错无法运行。这是因为同花顺的【问财】接口的限制。
- 如果用户准备数据时候，准备的时间区间太大导致预处理阶段生成的数据文件较大，策略可能无法正确读取文件。这是同花顺supermind平台的限制。
- 策略日志会很多，因为作者没优化，麻烦还请暂时忽略。
