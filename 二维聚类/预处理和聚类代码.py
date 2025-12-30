# 完整代码 - 修正版
import json
import pandas as pd
import csv
import numpy as np
import ast
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates
import warnings
import os
from datetime import datetime

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# base_path = r''
base_path = '.'  # 定义当前路径
# os.chdir(base_path)
warnings.filterwarnings('ignore')

# ------------------------------- 数据处理 -------------------------------
print("正在加载和处理数据...")

# 1. 读取游戏数据
games_json = json.load(open('raw_data/steam_2025_5k-dataset-games_20250831/steam_2025_5k-dataset-games_20250831.json'))
games = games_json['games']

# 2. 读取评论数据
review_json = json.load(open('raw_data/steam_2025_5k-dataset-reviews_20250901/steam_2025_5k-dataset-reviews_20250901.json', encoding='utf-8'))
reviews = review_json['reviews']

# 3. 处理评论数据，计算平均游玩时间
game_reviews = {}
for review in reviews:
    appid = review['appid']
    game_reviews[appid] = review['review_data']['query_summary']
    sum_time = 0
    if review['review_data']['reviews']:
        for r in review['review_data']['reviews']:
            sum_time += r["author"]['playtime_forever']
        sum_time = sum_time / len(review['review_data']['reviews'])
    game_reviews[appid]['avg_playtime'] = sum_time

# 4. 提取游戏数据并构建行
rows = []
for game in games:
    if 'data' not in game['app_details']:
        continue
    data = game['app_details']['data']
    if data['type'] != 'game':
        continue
    if 'playtest' in game['name_from_applist'].lower() or 'demo' in game['name_from_applist'].lower() or 'beta' in game['name_from_applist'].lower():
        continue
    
    row = {
        'steam_appid': data['steam_appid'],
        'name': game['name_from_applist'],
        'type': data['type'],
        'developers': ';'.join(data['developers']) if 'developers' in data else None,
        'publishers': ';'.join(data['publishers']) if 'publishers' in data else None,
        'release_date': 'Coming Soon' if data['release_date']['coming_soon'] is True
                        else (pd.to_datetime(data['release_date']['date'], format='mixed').strftime('%Y-%m-%d')) if  data['release_date']['date'] != '' else None,
        'platforms_windows': data['platforms']['windows'],
        'platforms_mac': data['platforms']['mac'],
        'platforms_linux': data['platforms']['linux'],
        'support_english': (1 if 'english' in data['supported_languages'].lower() else 0) if 'supported_languages' in data else None,
        'supported_languages': data['supported_languages'] if 'supported_languages' in data else None,
        'controller_support': data['controller_support'] if 'controller_support' in data else None,
        'categories': ';'.join([c['description'] for c in data['categories']]) if 'categories' in data else None,
        'genres': ';'.join([g['description'] for g in data['genres']]) if 'genres' in data else None,
        "achievements": data['achievements']['total'] if 'achievements' in data else None,
        'dlc': len(data['dlc']) if 'dlc' in data else None,
        'required_age': data['required_age'],
        'ratings': data['ratings'] if 'ratings' in data else None,
        'metacritic': data['metacritic'] if 'metacritic' in data else None,
        'reviews': data['reviews'] if 'reviews' in data else None,
        'recommendations': data['recommendations']['total'] if 'recommendations' in data else None,
        'total_reviews': game_reviews[data['steam_appid']]['total_reviews'] if data['steam_appid'] in game_reviews else None,
        'total_positive_reviews': game_reviews[data['steam_appid']]['total_positive'] if data['steam_appid'] in game_reviews else None,
        'total_negative_reviews': game_reviews[data['steam_appid']]['total_negative'] if data['steam_appid'] in game_reviews else None,
        'review_score': game_reviews[data['steam_appid']]['review_score'] if data['steam_appid'] in game_reviews else None,
        'review_score_desc': game_reviews[data['steam_appid']]['review_score_desc'] if data['steam_appid'] in game_reviews else None,
        'avg_playtime': game_reviews[data['steam_appid']]['avg_playtime'] if data['steam_appid'] in game_reviews else None, 
        'is_free': data['is_free'] if 'is_free' in data else None,
        'price_currency': data['price_overview']['currency'] if 'price_overview' in data else None,
        'price_initial': data['price_overview']['initial'] if 'price_overview' in data else 0,
        'price_final': data['price_overview']['final'] if 'price_overview' in data else 0,
        'price_discount_percent': data['price_overview']['discount_percent'] if 'price_overview' in data else 0,
    }
    rows.append(row)

print(f"共处理了 {len(rows)} 个游戏")

# 5. 写入临时CSV文件
with open('temp_steam_games.csv', 'w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    for row in rows:
        writer.writerow(row)

print("数据处理完成，正在加载到DataFrame...")

# ------------------------------- 数据清洗和预处理 -------------------------------
steam_df = pd.read_csv('temp_steam_games.csv')
cleaned_steam_df = steam_df.copy()

print(f"原始数据形状: {cleaned_steam_df.shape}")
print("开始数据清洗和预处理...")

# 1. 删除release_date为"Coming Soon"和空白缺失值的记录
print("处理release_date...")
original_count = len(cleaned_steam_df)

# 过滤掉"Coming Soon"和空值
cleaned_steam_df = cleaned_steam_df[
    (~cleaned_steam_df['release_date'].isna()) & 
    (cleaned_steam_df['release_date'] != 'Coming Soon') &
    (cleaned_steam_df['release_date'] != '')
]

deleted_count = original_count - len(cleaned_steam_df)
print(f"删除了 {deleted_count} 条release_date为'Coming Soon'或空值的记录")
print(f"剩余记录数: {len(cleaned_steam_df)}")

# 2. 生成"销售评论比"和"copies sold"属性
print("生成销售相关属性...")

def get_sales_review_ratio(release_date):
    """根据发行年份获取销售评论比"""
    try:
        # 解析日期格式，支持多种格式
        if pd.isna(release_date) or release_date == '':
            return None
        
        # 尝试不同日期格式
        date_formats = ['%Y-%m-%d', '%Y/%m/%d', '%Y.%m.%d', '%d %b, %Y', '%b %d, %Y']
        year = None
        
        for fmt in date_formats:
            try:
                dt = datetime.strptime(str(release_date), fmt)
                year = dt.year
                break
            except ValueError:
                continue
        
        if year is None:
            # 尝试直接提取年份
            import re
            year_match = re.search(r'(\d{4})', str(release_date))
            if year_match:
                year = int(year_match.group(1))
            else:
                return None
        
        # 根据年份确定销售评论比
        if year < 2014:
            return 60
        elif 2014 <= year <= 2016:
            return 50
        elif year == 2017:
            return 40
        elif year == 2018 or year == 2019:
            return 35
        elif year >= 2020:
            return 30
        else:
            return None
            
    except Exception as e:
        print(f"解析日期时出错: {release_date}, 错误: {e}")
        return None

# 添加销售评论比
cleaned_steam_df['sales_review_ratio'] = cleaned_steam_df['release_date'].apply(get_sales_review_ratio)

# 添加copies sold
cleaned_steam_df['copies_sold'] = cleaned_steam_df.apply(
    lambda row: row['total_reviews'] * row['sales_review_ratio'] 
    if pd.notna(row['sales_review_ratio']) and pd.notna(row['total_reviews']) 
    else 0,
    axis=1
)

# 创建好评率特征
cleaned_steam_df['positive_review_rate'] = cleaned_steam_df.apply(
    lambda row: row['total_positive_reviews'] / row['total_reviews'] if row['total_reviews'] > 0 else 0,
    axis=1
)

# 3. 填充缺失值
cleaned_steam_df['developers'].fillna('Unknown', inplace=True)
cleaned_steam_df['publishers'].fillna('Unknown', inplace=True)
cleaned_steam_df['support_english'].fillna('Unknown', inplace=True)
cleaned_steam_df['supported_languages'].fillna('', inplace=True)
cleaned_steam_df['supported_languages'] = cleaned_steam_df['supported_languages'].apply(
    lambda x: len(x.split(',')) if x != '' else 0
)
cleaned_steam_df['controller_support'].fillna('Unknown', inplace=True)
cleaned_steam_df['achievements'].fillna(0, inplace=True)
cleaned_steam_df['dlc'].fillna(0, inplace=True)
cleaned_steam_df['avg_playtime'].fillna(0, inplace=True)

# 4. 价格转换（先转换，再删除price_currency）
print("价格转换...")
exchange_rate = {
    'USD': 1.0,
    'EUR': 1.1690,
    'PHP': 0.0175,
    'AUD': 0.6543,
    'RUB': 0.0127,
    'CAD': 0.7278,
    'MXN': 0.0536,
    'PLN': 0.2741
}

cleaned_steam_df['price_currency'].fillna('USD', inplace=True)
cleaned_steam_df['price_initial'] = cleaned_steam_df['price_initial'] * cleaned_steam_df['price_currency'].map(exchange_rate)
cleaned_steam_df['price_final'] = cleaned_steam_df['price_final'] * cleaned_steam_df['price_currency'].map(exchange_rate)
cleaned_steam_df['price_initial'] = cleaned_steam_df['price_initial'].round(0).astype(int)
cleaned_steam_df['price_final'] = cleaned_steam_df['price_final'].round(0).astype(int)

# 5. 生成开发商、发行商01变量
print("生成开发商、发行商01变量...")
developer_counts = cleaned_steam_df['developers'].value_counts()
publisher_counts = cleaned_steam_df['publishers'].value_counts()

cleaned_steam_df['is_mainstream_developer'] = cleaned_steam_df['developers'].apply(
    lambda x: 1 if x != 'Unknown' and developer_counts.get(x, 0) > 1 else 0
)
cleaned_steam_df['is_mainstream_publisher'] = cleaned_steam_df['publishers'].apply(
    lambda x: 1 if x != 'Unknown' and publisher_counts.get(x, 0) > 1 else 0
)

# 6. One-hot encoding
support_english_dummies = pd.get_dummies(cleaned_steam_df['support_english'], prefix='support_english')
cleaned_steam_df['controller_support'] = cleaned_steam_df['controller_support'].replace({'full': 1, 'Unknown': 0})

# 7. 游戏类别编码
print("正在进行MultiHot编码...")
cleaned_steam_df['categories_list'] = cleaned_steam_df['categories'].fillna('').astype(str).apply(
    lambda x: x.split(';') if x != '' else []
)
categories_mlb = MultiLabelBinarizer()
categories_matrix = categories_mlb.fit_transform(cleaned_steam_df['categories_list'])
categories_df = pd.DataFrame(categories_matrix, columns=[f'category_{col}' for col in categories_mlb.classes_], index=cleaned_steam_df.index)
print(f"Categories编码后形状: {categories_df.shape}")

cleaned_steam_df['genres_list'] = cleaned_steam_df['genres'].fillna('').astype(str).apply(
    lambda x: x.split(';') if x != '' else []
)
genres_mlb = MultiLabelBinarizer()
genre_matrix = genres_mlb.fit_transform(cleaned_steam_df['genres_list'])
genre_df = pd.DataFrame(genre_matrix, columns=[f'genre_{col}' for col in genres_mlb.classes_], index=cleaned_steam_df.index)
print(f"Genres编码后形状: {genre_df.shape}")

# 将编码结果合并回cleaned_steam_df
print("将编码结果合并回主数据...")
cleaned_steam_df = pd.concat([cleaned_steam_df, categories_df, genre_df, support_english_dummies], axis=1)

# 8. 多语言冗余合并（保留英语）
print("处理多语言冗余列...")
language_merge_map = {
    'category_Un jugador': 'category_Single-player',
    'category_Для одного игрока': 'category_Single-player',
    'category_Compat. parcial con control': 'category_Full controller support',
    'category_Полная поддержка контроллеров': 'category_Full controller support',
    'category_Préstamo familiar': 'category_Family Sharing',
    'category_Семейный доступ': 'category_Family Sharing',
    'category_Logros de Steam': 'category_Steam Achievements',
    'category_Достижения Steam': 'category_Steam Achievements',
    'category_Remote Play para TV': 'category_Remote Play on TV',
    'category_Remote Play на телевизоре': 'category_Remote Play on TV',
    'category_Remote Play para móviles': 'category_Remote Play on Phone',
    'category_Remote Play на телефоне': 'category_Remote Play on Phone',
    'category_Remote Play para tabletas': 'category_Remote Play on Tablet',
    'category_Remote Play на планшете': 'category_Remote Play on Tablet',
    'category_Tarjetas de Steam': 'category_Steam Trading Cards',
    'genre_Acción': 'genre_Action',
    'genre_Экшены': 'genre_Action',
    'genre_Инди': 'genre_Indie',
    'genre_Deportes': 'genre_Sports',
    'genre_Carreras': 'genre_Racing',
}

# 只处理数据中实际存在的列
actual_columns = set(cleaned_steam_df.columns)
merge_map_filtered = {k: v for k, v in language_merge_map.items() if k in actual_columns}

for source_col, target_col in merge_map_filtered.items():
    if source_col in cleaned_steam_df.columns and target_col in cleaned_steam_df.columns:
        cleaned_steam_df[target_col] = cleaned_steam_df[[target_col, source_col]].max(axis=1)
        cleaned_steam_df.drop(columns=[source_col], inplace=True)

# 9. 删除不需要的列（在价格转换之后）
print("删除不需要的列...")
columns_to_drop = [
    'type', 'release_date', 'price_currency', 'sales_review_ratio',
    'total_reviews', 'total_positive_reviews', 'total_negative_reviews',
    'review_score', 'review_score_desc', 'support_english',
    'categories', 'genres', 'categories_list', 'genres_list',
    'reviews', 'metacritic', 'recommendations', 'ratings',
    'developers', 'publishers'  # 已经替换为01变量
]

columns_to_drop_existing = [col for col in columns_to_drop if col in cleaned_steam_df.columns]
cleaned_steam_df.drop(columns=columns_to_drop_existing, inplace=True)
print(f"删除了 {len(columns_to_drop_existing)} 个不需要的列")

# 保存预处理数据
preprocessed_file = 'steam预处理数据.csv'
cleaned_steam_df.to_csv(preprocessed_file, index=False, encoding='utf-8')
print(f"\n预处理数据已保存为 '{preprocessed_file}'")
print(f"最终数据形状: {cleaned_steam_df.shape}")

# ------------------------------- 聚类分析 -------------------------------
print("\n开始聚类分析...")

# 准备聚类特征：只用好评率和销量
cluster_features = ['copies_sold', 'positive_review_rate']
train_df = cleaned_steam_df[cluster_features].copy()

print(f"用于聚类的特征: {cluster_features}")
print(f"聚类数据形状: {train_df.shape}")

# 只对销量取对数并归一化
train_df['copies_sold'] = np.log1p(train_df['copies_sold'])
scaler = MinMaxScaler()
train_df['copies_sold'] = scaler.fit_transform(train_df[['copies_sold']])

# 定义聚类算法
algorithms = {
    'kmeans': {'name': 'KMeans', 'labels': None},
    'hierarchical': {'name': '层次聚类', 'labels': None},
    'dbscan': {'name': 'DBSCAN', 'labels': None}
}

# KMeans聚类
print("\n1. KMeans聚类...")
best_kmeans_k = 3
best_kmeans_silhouette = -1
best_kmeans_labels = None

for k in range(3, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(train_df)
    
    if len(np.unique(labels)) > 1:
        silhouette = silhouette_score(train_df, labels)
        print(f"  K={k}: 轮廓系数 = {silhouette:.4f}")
        
        if silhouette > best_kmeans_silhouette:
            best_kmeans_silhouette = silhouette
            best_kmeans_k = k
            best_kmeans_labels = labels

print(f"  KMeans最优K值: {best_kmeans_k}, 轮廓系数: {best_kmeans_silhouette:.4f}")
algorithms['kmeans']['labels'] = best_kmeans_labels
algorithms['kmeans']['k'] = best_kmeans_k
algorithms['kmeans']['silhouette'] = best_kmeans_silhouette

# 层次聚类
print("\n2. 层次聚类...")
best_hierarchical_silhouette = -1
best_hierarchical_labels = None

for linkage in ['ward', 'complete', 'average', 'single']:
    try:
        hierarchical = AgglomerativeClustering(n_clusters=best_kmeans_k, linkage=linkage)
        labels = hierarchical.fit_predict(train_df)
        
        if len(np.unique(labels)) > 1:
            silhouette = silhouette_score(train_df, labels)
            print(f"  {linkage}: 轮廓系数 = {silhouette:.4f}")
            
            if silhouette > best_hierarchical_silhouette:
                best_hierarchical_silhouette = silhouette
                best_hierarchical_labels = labels
    except Exception:
        continue

print(f"  层次聚类最优轮廓系数: {best_hierarchical_silhouette:.4f}")
algorithms['hierarchical']['labels'] = best_hierarchical_labels
algorithms['hierarchical']['k'] = best_kmeans_k
algorithms['hierarchical']['silhouette'] = best_hierarchical_silhouette

# DBSCAN聚类
print("\n3. DBSCAN聚类...")
best_dbscan_silhouette = -1
best_dbscan_labels = None
best_dbscan_n_clusters = 0

eps_values = [0.1, 0.2, 0.3, 0.4, 0.5]
min_samples_values = [5, 10, 15]

for eps in eps_values:
    for min_samples in min_samples_values:
        try:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(train_df)
            
            # 排除噪声点
            valid_mask = labels != -1
            if np.sum(valid_mask) < len(labels) * 0.5:
                continue
                
            valid_labels = labels[valid_mask]
            valid_data = train_df[valid_mask]
            
            n_clusters = len(np.unique(valid_labels))
            if n_clusters < 2 or n_clusters > 10:
                continue
                
            if len(np.unique(valid_labels)) > 1:
                silhouette = silhouette_score(valid_data, valid_labels)
                print(f"  eps={eps}, min_samples={min_samples}: {n_clusters}个簇, 轮廓系数 = {silhouette:.4f}")
                
                if silhouette > best_dbscan_silhouette:
                    best_dbscan_silhouette = silhouette
                    best_dbscan_n_clusters = n_clusters
                    best_dbscan_labels = labels
        except Exception:
            continue

# print(f"  DBSCAN最优轮廓系数: {best_dbscan_silhouette:.4f}")
# algorithms['dbscan']['labels'] = best_dbscan_labels
# algorithms['dbscan']['silhouette'] = best_dbscan_silhouette
# algorithms['dbscan']['k'] = best_dbscan_n_clusters

import seaborn as sns

import seaborn as sns
from sklearn.decomposition import PCA

# --- 环境设置 ---
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei'] 
plt.rcParams['axes.unicode_minus'] = False

# # ------------------------------- 1. 参数敏感性分析 (曲线图) -------------------------------
# print("\n正在生成参数敏感性分析图...")

# # 收集 KMeans 数据
# kmeans_curve = []
# for k in range(3, 11):
#     km = KMeans(n_clusters=k, random_state=42, n_init=10)
#     l = km.fit_predict(train_df)
#     kmeans_curve.append({'K': k, 'Silhouette': silhouette_score(train_df, l)})

# # 收集 层次聚类 数据
# hier_curve = []
# for link in ['ward', 'complete', 'average', 'single']:
#     hc = AgglomerativeClustering(n_clusters=best_kmeans_k, linkage=link)
#     l = hc.fit_predict(train_df)
#     hier_curve.append({'Linkage': link, 'Silhouette': silhouette_score(train_df, l)})

# # 收集 DBSCAN 数据 (固定 min_samples=10，观察 eps 变化)
# dbscan_curve = []
# for e in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]:
#     db = DBSCAN(eps=e, min_samples=10)
#     l = db.fit_predict(train_df)
#     if len(np.unique(l[l != -1])) > 1:
#         s = silhouette_score(train_df[l != -1], l[l != -1])
#         dbscan_curve.append({'eps': e, 'Silhouette': s})

# # 绘制参数对比三合一图
# fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# # KMeans 曲线
# sns.lineplot(ax=axes[0], data=pd.DataFrame(kmeans_curve), x='K', y='Silhouette', marker='o', color='#2c3e50', linewidth=2)
# axes[0].axvline(x=best_kmeans_k, color='#e74c3c', linestyle='--', label=f'Best K={best_kmeans_k}')
# axes[0].set_title('KMeans 聚类数与轮廓系数', fontsize=13)
# axes[0].legend()

# # 层次聚类 柱状图
# sns.barplot(ax=axes[1], data=pd.DataFrame(hier_curve), x='Linkage', y='Silhouette', palette='flare')
# axes[1].set_title('层次聚类 不同Linkage表现', fontsize=13)

# # DBSCAN 曲线
# sns.lineplot(ax=axes[2], data=pd.DataFrame(dbscan_curve), x='eps', y='Silhouette', marker='s', color='#d35400', linewidth=2)
# axes[2].set_title('DBSCAN eps半径与轮廓系数', fontsize=13)

# plt.tight_layout()
# plt.savefig(os.path.join(base_path, '00_参数敏感性分析对比.png'), dpi=300)
# plt.show()

# ------------------------------- 2. 各算法详细可视化循环 -------------------------------
print("\n正在生成各算法详细分析图...")

comparison_data = []

for algo_key, algo_info in algorithms.items():
    if algo_info['labels'] is None:
        continue
    
    # 记录用于最终对比的数据
    comparison_data.append({'Algorithm': algo_info['name'], 'Silhouette': algo_info.get('silhouette', 0)})
    
    # 路径准备
    algo_dir = os.path.join(base_path, algo_key)
    viz_dir = os.path.join(algo_dir, '可视化')
    if not os.path.exists(viz_dir): os.makedirs(viz_dir)

    # # --- A. PCA 旋转空间图 (flare 配色) ---
    # pca = PCA(n_components=2)
    # X_pca = pca.fit_transform(train_df)
    # plt.figure(figsize=(9, 6))
    # sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=algo_info['labels'], 
    #                 palette='flare', s=60, alpha=0.8, edgecolor='w', legend='full')
    # plt.title(f'{algo_info["name"]} - PCA 投影 (寻找最大方差方向)', fontsize=14)
    # plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    # plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    # plt.savefig(os.path.join(viz_dir, '01_PCA旋转空间图.png'), dpi=300, bbox_inches='tight')
    # plt.close()

    # # --- B. 业务分布图 (mako 配色) ---
    # plt.figure(figsize=(9, 6))
    # sns.scatterplot(x=train_df['copies_sold'], y=train_df['positive_review_rate'], 
    #                 hue=algo_info['labels'], palette='mako', s=60, alpha=0.7, edgecolor='w')
    # plt.title(f'{algo_info["name"]} - 销量与好评率分布', fontsize=14)
    # plt.xlabel('估算销量 (Log标准化后)')
    # plt.ylabel('好评率')
    # plt.savefig(os.path.join(viz_dir, '02_业务分布图.png'), dpi=300, bbox_inches='tight')
    # plt.close()

    # --- C. 特征热力图 (rocket_r 配色) ---
    plt.figure(figsize=(7, 4))
    temp_df = train_df.copy()
    temp_df['cluster'] = algo_info['labels']
    
    # 核心步骤：计算均值矩阵并打印
    cluster_means = temp_df.groupby('cluster').mean()
    
    # 【新增：打印数据到控制台】
    print(f"\n>>> {algo_info['name']} 各簇中心特征强度数值表:")
    print("-" * 50)
    print(cluster_means.round(3)) # 保留3位小数打印
    print("-" * 50)
    
    sns.heatmap(cluster_means, annot=True, cmap='rocket_r', fmt='.3f', cbar_kws={'label': '特征强度'})
    plt.title(f'{algo_info["name"]} - 簇中心特征强度', fontsize=12)
    plt.savefig(os.path.join(viz_dir, '03_特征热力图.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # --- D. 数据导出 (Excel) ---
    clustered_df = cleaned_steam_df.copy()
    clustered_df.insert(1, 'cluster_label', algo_info['labels'])
    clustered_df.to_csv(os.path.join(algo_dir, f'steam聚类结果_{algo_key}.csv'), index=False, encoding='utf-8-sig')

# ------------------------------- 3. 算法全家福总对比 -------------------------------
if comparison_data:
    plt.figure(figsize=(10, 6))
    comp_df = pd.DataFrame(comparison_data)
    ax = sns.barplot(x='Algorithm', y='Silhouette', data=comp_df, palette='viridis')
    
    # 加上数值标注
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.4f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=12, fontweight='bold', xytext=(0, 5), textcoords='offset points')
    
    plt.title('各聚类算法最终轮廓系数对比', fontsize=15)
    plt.ylim(0, 1.0)
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(base_path, '04_算法性能全家福.png'), dpi=300, bbox_inches='tight')
    plt.show()

print("\n所有图表已生成：")
print("1. [00_参数敏感性分析对比.png] - 证明参数选择的合理性")
print("2. 各算法文件夹/可视化/ - 包含PCA图、业务图和热力图")
print("3. [04_算法性能对比图.png] - 最终效果对比图")