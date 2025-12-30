# Steam 数据处理流程说明文档

## 项目概述

本项目提供两个独立的Python脚本，用于将3份Steam相关数据集处理为RapidMiner兼容格式，供组员进行数据挖掘建模使用。

## 数据集说明

### 输入数据文件

1. **steam_.csv** (主表)
   - 记录数: 27,075 条
   - 核心属性: 18 个
   - 主要字段: `appid`, `name`, `release_date`, `genres`, `positive_ratings`, `negative_ratings`, `average_playtime`, `owners`, `price` 等
   - 内容: 游戏元数据和市场表现数据

2. **steam_games.csv** (游戏详情表)
   - 记录数: 7,970 条
   - 属性数: 125+ 个
   - 主要字段: `appid`, `short_description`, `pc_requirements`, `rating_esrb_rating`, `developers`, `publishers` 等
   - 内容: 游戏详细信息补充

3. **steam_reviews.csv** (评论表)
   - 记录数: 37,778 条
   - 主要字段: `appid`, `review`, `voted_up`, `timestamp_created`, `author_steamid` 等
   - 内容: 用户评论数据

### 关联字段

所有数据表通过 **`appid`** (应用ID) 字段进行关联。

## 处理流程

### Step 1: 基础拼接 (step1_basic_merge.py)

**目标**: 完成数据排布和基础拼接，不做深度清洗，快速提供结构规整的基础数据文件。

**核心任务**:
1. ✅ 基于 `appid` 字段，将3份数据进行左连接（以 `steam_.csv` 为主表）
2. ✅ 清理列名（去除特殊字符和空格，确保RapidMiner兼容）
3. ✅ 聚合评论数据（按 `appid` 统计评论总数、好评率等）
4. ✅ 标记缺失值为 "NA"（不做填充）
5. ✅ 删除多余的 Unnamed 列
6. ✅ 输出 UTF-8 编码的 CSV 文件

**输出文件**: `steam_rapidminer_basic.csv`

**特点**:
- 保留原始数据，不做修改
- 缺失值统一标记为 "NA"
- 列名已清理，符合RapidMiner格式要求
- 可直接导入RapidMiner查看数据结构

### Step 2: 深度清洗预处理 (step2_data_preprocessing.py)

**目标**: 基于Step1的输出文件，完成RapidMiner建模所需的数据清洗，确保数据质量达标。

**核心任务**:

1. **缺失值处理**
   - 数值型字段: 使用均值或中位数填充（偏态分布用中位数，正态分布用均值）
   - 分类型字段: 使用 "Unknown" 填充
   - 布尔型字段: 使用 0 填充

2. **数据类型标准化**
   - 数值型字段: 统一转为 `int`/`float`，剔除文本杂质
   - 日期字段: 转为 `YYYY-MM-DD` 标准格式
   - 布尔型字段: 转为 0/1 编码
   - 多值字段: 统一使用分号 (`;`) 分隔

3. **异常值处理**
   - 使用 IQR (四分位距) 方法检测异常值
   - 对数值型字段（如 `playtime`, `ratings`）进行异常值处理
   - 将极端异常值替换为合理边界值

4. **特征规整**
   - 多值字段（`genres`, `categories` 等）统一格式
   - 去除字符串首尾空格
   - 删除完全重复的行

**输出文件**: `steam_rapidminer_preprocessed.csv`

**特点**:
- 数据质量达标，可直接用于建模
- 所有缺失值已填充
- 数据类型正确识别
- 异常值已处理
- 符合RapidMiner建模要求

## 使用方法

### 环境要求

- Python 3.6+
- pandas
- numpy

安装依赖:
```bash
pip install pandas numpy
```

### 执行步骤

#### Step 1: 基础拼接

```bash
python step1_basic_merge.py
```

**输出**: `steam_rapidminer_basic.csv`

**预期结果**:
- 数据已合并（左连接）
- 列名已清理
- 缺失值标记为 "NA"
- 可直接导入RapidMiner查看

#### Step 2: 深度清洗

```bash
python step2_data_preprocessing.py
```

**输入**: `steam_rapidminer_basic.csv` (Step1的输出)

**输出**: `steam_rapidminer_preprocessed.csv`

**预期结果**:
- 缺失值已填充
- 数据类型已标准化
- 异常值已处理
- 可直接用于RapidMiner建模

## RapidMiner 导入说明

### 文件格式要求

- **编码**: UTF-8
- **分隔符**: 逗号 (`,`)
- **表头**: 第一行
- **列名**: 无特殊字符，无空格（已自动处理）

### 导入步骤

1. 打开 RapidMiner Studio
2. 创建新的 Process
3. 添加 "Read CSV" 操作符
4. 设置文件路径为 `steam_rapidminer_preprocessed.csv`
5. 设置编码为 UTF-8
6. 设置分隔符为逗号
7. 勾选 "First row is column names"
8. 运行导入

### 数据类型识别

RapidMiner 会自动识别数据类型：
- **数值型**: `price`, `positive_ratings`, `average_playtime` 等
- **分类型**: `name`, `genres`, `categories` 等
- **二分类**: `is_free`, `voted_up`, `platform_windows` 等（已编码为 0/1）
- **日期型**: `release_date`, `timestamp_created` 等（格式: YYYY-MM-DD）

## 建模建议

### 贝叶斯分类

**适用场景**: 预测游戏是否受欢迎、是否免费等

**建议特征**:
- 数值型: `price`, `positive_ratings`, `average_playtime`
- 分类型: `genres`, `categories`, `developers`
- 目标变量: `is_free` (0/1), `reviews_positive_rate` (离散化)

**注意事项**:
- 确保目标变量为分类型
- 数值型特征可进行离散化处理

### 决策树

**适用场景**: 分析影响游戏受欢迎程度的关键因素

**建议特征**:
- 所有数值型和分类型特征
- 目标变量: `positive_ratings` (连续值) 或 `reviews_positive_rate` (离散化)

**注意事项**:
- 处理多值字段（如 `genres`）时，可考虑拆分或使用频次统计
- 确保无异常值（已在Step2中处理）

### 关联规则挖掘

**适用场景**: 发现游戏特征之间的关联关系

**建议特征**:
- 离散化的数值变量: `price_range`, `playtime_range`
- 多值字段: `genres`, `categories`, `platforms`
- 二分类变量: `is_free`, `platform_windows`, `platform_mac`

**注意事项**:
- 需要将连续值离散化为分类变量
- 多值字段需要拆分或使用集合表示
- 确保数据为事务型格式

## 数据质量检查

### Step1 输出检查

- [ ] 数据行数是否正确（应与主表一致: 27,075 行）
- [ ] 列名是否无特殊字符
- [ ] 缺失值是否标记为 "NA"
- [ ] 无 Unnamed 列

### Step2 输出检查

- [ ] 缺失值是否已填充
- [ ] 数值型字段是否为数值类型
- [ ] 日期字段格式是否为 YYYY-MM-DD
- [ ] 布尔型字段是否为 0/1
- [ ] 多值字段是否使用分号分隔
- [ ] 异常值是否已处理

## 常见问题

### Q1: Step1 运行时提示文件不存在

**解决方案**: 确保以下文件在同一目录下:
- `steam_.csv`
- `steam_games.csv`
- `steam_reviews.csv`

### Q2: Step2 运行时提示找不到输入文件

**解决方案**: 确保先运行 Step1，生成 `steam_rapidminer_basic.csv` 文件

### Q3: RapidMiner 导入时出现编码错误

**解决方案**: 
- 确认文件编码为 UTF-8
- 在 RapidMiner 中设置编码为 UTF-8
- 如果仍有问题，可尝试 GBK 编码（需要修改脚本中的编码设置）

### Q4: 数据量太大，处理时间过长

**解决方案**:
- Step1 通常较快（1-2分钟）
- Step2 可能需要较长时间（5-10分钟），请耐心等待
- 可以先用小样本测试

### Q5: 某些字段在建模时无法识别

**解决方案**:
- 检查字段类型是否正确
- 在 RapidMiner 中使用 "Set Role" 操作符设置字段角色
- 对于多值字段，考虑先进行拆分或频次统计

## 文件清单

```
项目目录/
├── step1_basic_merge.py          # Step1: 基础拼接脚本
├── step2_data_preprocessing.py   # Step2: 深度清洗脚本
├── README_数据处理流程.md        # 本文档
├── steam_.csv                    # 输入: 主表数据
├── steam_games.csv               # 输入: 游戏详情数据
├── steam_reviews.csv             # 输入: 评论数据
├── steam_rapidminer_basic.csv    # 输出: Step1结果（中间文件）
└── steam_rapidminer_preprocessed.csv  # 输出: Step2结果（最终文件）
```

## 更新日志

- 2025-01-XX: 初始版本，实现两步数据处理流程
  - Step1: 基础拼接和格式适配
  - Step2: 深度清洗和预处理

## 联系方式

如有问题，请联系项目负责人或查看代码注释。

