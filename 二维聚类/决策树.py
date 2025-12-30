import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import subprocess
import sys
from sklearn.model_selection import train_test_split, cross_validate, KFold

# 1. 检查并安装必要的库
print("检查并安装必要的库...")
required_packages = ['xgboost', 'scikit-learn']

for package in required_packages:
    try:
        if package == 'xgboost':
            import xgboost
            print(f"✓ {package} 已安装")
        elif package == 'scikit-learn':
            import sklearn
            print(f"✓ {package} 已安装")
    except ImportError:
        print(f"正在安装 {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✓ {package} 安装完成")

# 现在导入所有需要的模块
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# 设置工作目录
base_path = r'E:\大数据\steam数据挖掘'
os.chdir(base_path)

# 创建决策树文件夹
decision_tree_dir = os.path.join(base_path, '决策树分析')
if not os.path.exists(decision_tree_dir):
    os.makedirs(decision_tree_dir)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 1. 读取聚类结果数据
print("读取聚类结果数据...")
# 读取kmeans聚类结果 - 尝试多种编码
cluster_data_path = os.path.join(base_path, 'kmeans', 'steam聚类_kmeans.csv')

# 尝试不同的编码方式读取文件
encodings_to_try = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'latin1', 'cp1252']

df = None
for encoding in encodings_to_try:
    try:
        print(f"尝试用 {encoding} 编码读取文件...")
        df = pd.read_csv(cluster_data_path, encoding=encoding)
        print(f"✓ 成功用 {encoding} 编码读取文件")
        break
    except UnicodeDecodeError as e:
        print(f"  {encoding} 编码失败: {str(e)[:50]}...")
    except Exception as e:
        print(f"  {encoding} 编码失败: {str(e)[:50]}...")

if df is None:
    # 如果所有编码都失败，尝试用二进制读取然后修复
    print("所有编码都失败，尝试二进制读取并修复...")
    try:
        with open(cluster_data_path, 'rb') as f:
            content = f.read()
        
        # 尝试解码并替换非法字符
        for encoding in encodings_to_try:
            try:
                decoded = content.decode(encoding, errors='replace')
                df = pd.read_csv(pd.compat.StringIO(decoded))
                print(f"✓ 成功用 {encoding} 编码（替换非法字符）读取文件")
                break
            except:
                continue
    except Exception as e:
        print(f"二进制读取也失败: {e}")
        raise

if df is None:
    raise ValueError("无法读取文件，请检查文件编码")

print(f"数据形状: {df.shape}")
print(f"列名: {df.columns.tolist()}")
print(f"聚类标签分布:\n{df['cluster_label'].value_counts().sort_index()}")

# 2. 准备特征和标签
print("\n准备特征和标签...")
# 移除标识列、聚类标签列和两个聚类指标列
exclude_columns = ['steam_appid', 'name', 'cluster_label', 'copies_sold', 'positive_review_rate']

# 确保要排除的列在数据中存在
existing_exclude_columns = [col for col in exclude_columns if col in df.columns]
print(f"要排除的列: {existing_exclude_columns}")

X = df.drop(columns=existing_exclude_columns, axis=1)
y = df['cluster_label']

print(f"特征数量: {X.shape[1]}")
print(f"样本数量: {X.shape[0]}")
print(f"前10个特征列: {X.columns.tolist()[:10]}")

# 3. 划分训练集和测试集（9:1）
print("\n划分训练集和测试集（9:1）...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y
)
print(f"训练集大小: {X_train.shape}")
print(f"测试集大小: {X_test.shape}")

# 4. 随机森林参数调优
print("\n" + "="*80)
print("随机森林参数调优")
print("="*80)

# 定义参数组合（从限制复杂度到放松复杂度）
param_combinations = [
    # 组合1-4：限制复杂度（严格）
    {
        'name': '组合1-最严格',
        'n_estimators': 50,
        'max_depth': 5,
        'min_samples_split': 20,
        'min_samples_leaf': 10,
        'max_features': 0.3,
        'bootstrap': True
    },
    {
        'name': '组合2-较严格',
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 10,
        'min_samples_leaf': 5,
        'max_features': 0.5,
        'bootstrap': True
    },
    {
        'name': '组合3-中等',
        'n_estimators': 150,
        'max_depth': 15,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'max_features': 0.7,
        'bootstrap': True
    },
    {
        'name': '组合4-较放松',
        'n_estimators': 200,
        'max_depth': 20,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'max_features': 'sqrt',
        'bootstrap': True
    },
    # 组合5-8：放松复杂度（更自由）
    {
        'name': '组合5-放松',
        'n_estimators': 200,
        'max_depth': 30,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'max_features': None,
        'bootstrap': True
    },
    {
        'name': '组合6-更放松',
        'n_estimators': 250,
        'max_depth': 40,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'max_features': None,
        'bootstrap': True
    },
    {
        'name': '组合7-很放松',
        'n_estimators': 250,
        'max_depth': 50,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'max_features': None,
        'bootstrap': True
    },
    {
        'name': '组合8-最放松',
        'n_estimators': 300,
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'max_features': None,
        'bootstrap': True
    }
]

print("进行3折交叉验证参数调优...")
param_results = []

for i, params in enumerate(param_combinations):
    print(f"\n测试参数组合 {i+1}: {params['name']}")
    print(f"  n_estimators: {params['n_estimators']}")
    print(f"  max_depth: {params['max_depth']}")
    print(f"  min_samples_split: {params['min_samples_split']}")
    print(f"  min_samples_leaf: {params['min_samples_leaf']}")
    print(f"  max_features: {params['max_features']}")
    
    # 创建随机森林模型
    rf = RandomForestClassifier(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        min_samples_split=params['min_samples_split'],
        min_samples_leaf=params['min_samples_leaf'],
        max_features=params['max_features'],
        bootstrap=params['bootstrap'],
        random_state=42,
        n_jobs=-1
    )
    
    # 3折交叉验证
    cv = KFold(n_splits=3, shuffle=True, random_state=42)
    cv_results = cross_validate(
        rf, X_train, y_train,
        cv=cv,
        scoring='accuracy',
        return_train_score=True,
        n_jobs=-1
    )
    
    # 计算平均准确率
    avg_train_accuracy = np.mean(cv_results['train_score'])
    avg_test_accuracy = np.mean(cv_results['test_score'])
    avg_performance = (avg_train_accuracy + avg_test_accuracy) / 2
    
    param_results.append({
        'param_name': params['name'],
        'param_index': i+1,
        'avg_train_accuracy': avg_train_accuracy,
        'avg_test_accuracy': avg_test_accuracy,
        'avg_performance': avg_performance,
        'params': params
    })
    
    print(f"  平均训练准确率: {avg_train_accuracy:.4f}")
    print(f"  平均测试准确率: {avg_test_accuracy:.4f}")
    print(f"  平均性能: {avg_performance:.4f}")

# 5. 选择最优参数
print("\n" + "="*80)
print("参数调优结果")
print("="*80)

param_results_df = pd.DataFrame(param_results).sort_values('avg_performance', ascending=False)
print("\n参数组合性能排序:")
for i, row in param_results_df.iterrows():
    print(f"{row['param_name']}: 性能={row['avg_performance']:.4f} (训练={row['avg_train_accuracy']:.4f}, 测试={row['avg_test_accuracy']:.4f})")

# 选择最优参数
best_params = param_results_df.iloc[0]['params']
print(f"\n最优参数组合: {best_params['name']}")
print(f"最优参数: n_estimators={best_params['n_estimators']}, max_depth={best_params['max_depth']}, "
      f"min_samples_split={best_params['min_samples_split']}, min_samples_leaf={best_params['min_samples_leaf']}, "
      f"max_features={best_params['max_features']}")

# 6. 可视化参数调优结果
print("\n生成参数调优可视化图表...")
plt.figure(figsize=(14, 8))

# 准备数据
param_names = [f"组合{i+1}" for i in range(len(param_results))]
train_accuracies = [r['avg_train_accuracy'] for r in param_results]
test_accuracies = [r['avg_test_accuracy'] for r in param_results]
performances = [r['avg_performance'] for r in param_results]

x = np.arange(len(param_names))
width = 0.25

plt.bar(x - width, train_accuracies, width, label='平均训练准确率', alpha=0.8, color='skyblue')
plt.bar(x, test_accuracies, width, label='平均测试准确率', alpha=0.8, color='lightcoral')
plt.bar(x + width, performances, width, label='平均性能', alpha=0.8, color='lightgreen')

plt.xlabel('参数组合（从限制到放松）', fontsize=12)
plt.ylabel('准确率', fontsize=12)
plt.title('随机森林参数调优结果（3折交叉验证）', fontsize=16, fontweight='bold')
plt.xticks(x, param_names, rotation=45)
plt.ylim([0, 1.05])
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()

param_tuning_chart_path = os.path.join(decision_tree_dir, '随机森林参数调优结果.png')
plt.savefig(param_tuning_chart_path, dpi=300)
plt.show()

# 保存参数调优结果
param_tuning_df = pd.DataFrame(param_results)
param_tuning_file = os.path.join(decision_tree_dir, '随机森林参数调优结果.csv')
param_tuning_df.to_csv(param_tuning_file, index=False, encoding='utf-8')
print(f"参数调优结果已保存到: {param_tuning_file}")

# 7. 使用最优参数训练正式模型
print("\n" + "="*80)
print("使用最优参数训练正式模型")
print("="*80)

# 定义模型（尽可能使用最优参数）
models = {
    'CART': DecisionTreeClassifier(
        criterion='gini', 
        random_state=42,
        max_depth=best_params['max_depth'],  # 使用最优的max_depth
        min_samples_split=best_params['min_samples_split'],  # 使用最优的min_samples_split
        min_samples_leaf=best_params['min_samples_leaf'],  # 使用最优的min_samples_leaf
        max_features=best_params['max_features'],  # 使用最优的max_features
        min_impurity_decrease=0.0
    ),
    'C4.5': DecisionTreeClassifier(
        criterion='entropy', 
        random_state=42,
        max_depth=best_params['max_depth'],  # 使用最优的max_depth
        min_samples_split=best_params['min_samples_split'],  # 使用最优的min_samples_split
        min_samples_leaf=best_params['min_samples_leaf'],  # 使用最优的min_samples_leaf
        max_features=best_params['max_features'],  # 使用最优的max_features
        min_impurity_decrease=0.0
    ),
    'RandomForest': RandomForestClassifier(
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        min_samples_split=best_params['min_samples_split'],
        min_samples_leaf=best_params['min_samples_leaf'],
        max_features=best_params['max_features'],
        bootstrap=best_params['bootstrap'],
        random_state=42,
        n_jobs=-1
    ),
    'GradientBoosting': GradientBoostingClassifier(
        n_estimators=best_params['n_estimators'],  # 使用最优的n_estimators
        random_state=42,
        max_depth=best_params['max_depth'],  # 使用最优的max_depth
        min_samples_split=best_params['min_samples_split'],  # 使用最优的min_samples_split
        min_samples_leaf=best_params['min_samples_leaf'],  # 使用最优的min_samples_leaf
        learning_rate=0.1,  # 保持原设置
        subsample=1.0,  # 保持原设置
        max_features=best_params['max_features']  # 使用最优的max_features
    ),
    'XGBoost': XGBClassifier(
        n_estimators=best_params['n_estimators'],  # 使用最优的n_estimators
        random_state=42,
        max_depth=best_params['max_depth'] if best_params['max_depth'] is not None else 10,  # 使用最优的max_depth，如果为None则用10
        learning_rate=0.1,  # 保持原设置
        subsample=1.0,  # 保持原设置
        colsample_bytree=1.0 if best_params['max_features'] is None else best_params['max_features'],  # 使用最优的max_features
        min_child_weight=best_params['min_samples_leaf'],  # 使用最优的min_samples_leaf作为min_child_weight
        gamma=0,  # 保持原设置
        reg_alpha=0,  # 保持原设置
        reg_lambda=0,  # 保持原设置
        use_label_encoder=False, 
        eval_metric='mlogloss',
        n_jobs=-1
    )
}

# 8. 训练和评估正式模型
print("\n训练和评估正式模型...")
results = {}
for name, model in models.items():
    print(f"训练 {name}...")
    
    # 训练模型
    model.fit(X_train, y_train)
    
    # 预测
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # 计算准确率
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    model_performance = (train_accuracy + test_accuracy) / 2  # 计算性能指标
    
    results[name] = {
        'model': model,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'model_performance': model_performance,  # 添加性能指标
        'train_predictions': y_pred_train,
        'test_predictions': y_pred_test
    }
    
    print(f"  {name}: 训练准确率={train_accuracy:.4f}, 测试准确率={test_accuracy:.4f}, 性能={model_performance:.4f}")

# 9. 性能对比 - 使用(训练准确率+测试准确率)/2排序
print("\n模型性能对比（按(训练准确率+测试准确率)/2排序）:")
performance_df = pd.DataFrame([
    {'Model': name, 
     'Train_Accuracy': results[name]['train_accuracy'],
     'Test_Accuracy': results[name]['test_accuracy'],
     'Performance': results[name]['model_performance']}  # 性能指标
    for name in results.keys()
]).sort_values('Performance', ascending=False)  # 按性能指标排序

print(performance_df.to_string(index=False))
performance_df.to_csv(os.path.join(decision_tree_dir, '模型性能对比.csv'), index=False, encoding='utf-8')
print(f"\n性能对比已保存到: {os.path.join(decision_tree_dir, '模型性能对比.csv')}")

# 10. 提取最优3个模型的完整规则（按性能指标排名）
print("\n提取最优3个模型的完整规则...")
top_3_models = performance_df.head(3)['Model'].tolist()
print(f"最优模型（按性能排名）: {top_3_models}")
feature_names = X.columns.tolist()

for model_name in top_3_models:
    model = results[model_name]['model']
    rule_file = os.path.join(decision_tree_dir, f'{model_name}_分类规则.txt')
    
    print(f"\n处理 {model_name}...")
    
    with open(rule_file, 'w', encoding='utf-8') as f:
        f.write(f"模型: {model_name}\n")
        f.write(f"训练准确率: {results[model_name]['train_accuracy']:.4f}\n")
        f.write(f"测试准确率: {results[model_name]['test_accuracy']:.4f}\n")
        f.write(f"性能指标（训练+测试）/2: {results[model_name]['model_performance']:.4f}\n")
        f.write(f"性能排名: {top_3_models.index(model_name) + 1}\n")
        f.write(f"聚类标签: {sorted(df['cluster_label'].unique())}\n")
        f.write(f"训练样本数量: {len(X_train)}\n")
        f.write(f"测试样本数量: {len(X_test)}\n")
        
        if model_name == 'RandomForest':
            f.write(f"最优参数: n_estimators={best_params['n_estimators']}, max_depth={best_params['max_depth']}\n")
            f.write(f"          min_samples_split={best_params['min_samples_split']}, min_samples_leaf={best_params['min_samples_leaf']}\n")
            f.write(f"          max_features={best_params['max_features']}\n")
        
        f.write("="*120 + "\n\n")
        
        if model_name in ['CART', 'C4.5']:
            # 单棵树：提取完整规则
            print(f"  提取完整决策树规则...")
            tree_rules = export_text(model, feature_names=feature_names)
            f.write("完整决策树规则:\n")
            f.write("="*120 + "\n\n")
            f.write(tree_rules)
            print(f"  规则长度: {len(tree_rules)} 字符")
            
        elif model_name == 'RandomForest':
            # 随机森林：特征重要性和第一棵树的规则
            print(f"  提取特征重要性...")
            importances = model.feature_importances_
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            f.write("特征重要性（前50个特征）:\n")
            f.write("="*120 + "\n")
            for _, row in importance_df.head(50).iterrows():
                f.write(f"{row['feature']}: {row['importance']:.6f}\n")
            
            # 保存特征重要性
            importance_df.to_csv(os.path.join(decision_tree_dir, f'{model_name}_特征重要性.csv'), index=False, encoding='utf-8')
            
            print(f"  提取第一棵树的完整规则...")
            f.write(f"\n\n第一棵树的完整规则（共{len(model.estimators_)}棵树）:\n")
            f.write("="*120 + "\n")
            
            # 提取第一棵树
            first_tree = model.estimators_[0]
            f.write(f"\n{'='*60} 第一棵树 {'='*60}\n")
            tree_rules = export_text(first_tree, feature_names=feature_names)
            f.write(tree_rules)
            print(f"  规则长度: {len(tree_rules)} 字符")
            
        elif model_name == 'GradientBoosting':
            # GBDT：特征重要性和正中间树的规则
            print(f"  提取特征重要性...")
            importances = model.feature_importances_
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            f.write("特征重要性（前50个特征）:\n")
            f.write("="*120 + "\n")
            for _, row in importance_df.head(50).iterrows():
                f.write(f"{row['feature']}: {row['importance']:.6f}\n")
            
            # 保存特征重要性
            importance_df.to_csv(os.path.join(decision_tree_dir, f'{model_name}_特征重要性.csv'), index=False, encoding='utf-8')
            
            print(f"  提取正中间树的完整规则...")
            f.write(f"\n\n正中间树的完整规则（共{len(model.estimators_)}棵树）:\n")
            f.write("="*120 + "\n")
            
            # 计算正中间的树索引
            total_trees = len(model.estimators_)
            middle_tree_index = total_trees // 2
            
            # 提取正中间的树
            if total_trees > 0:
                middle_tree = model.estimators_[middle_tree_index, 0]
                f.write(f"\n{'='*60} 正中间树（第{middle_tree_index + 1}棵，共{total_trees}棵） {'='*60}\n")
                tree_rules = export_text(middle_tree, feature_names=feature_names)
                f.write(tree_rules)
                print(f"  提取第{middle_tree_index + 1}棵树的规则，规则长度: {len(tree_rules)} 字符")
            else:
                f.write("\n没有可用的树\n")
                print("  没有可用的树")
            
        elif model_name == 'XGBoost':
            # XGBoost：特征重要性和正中间树的规则
            print(f"  提取特征重要性...")
            importances = model.feature_importances_
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            f.write("特征重要性（前50个特征）:\n")
            f.write("="*120 + "\n")
            for _, row in importance_df.head(50).iterrows():
                f.write(f"{row['feature']}: {row['importance']:.6f}\n")
            
            # 保存特征重要性
            importance_df.to_csv(os.path.join(decision_tree_dir, f'{model_name}_特征重要性.csv'), index=False, encoding='utf-8')
            
            # 获取所有树的完整结构
            print(f"  提取正中间树的完整结构...")
            booster = model.get_booster()
            tree_dump = booster.get_dump()
            
            f.write(f"\n\n正中间树的完整结构（共{len(tree_dump)}棵树）:\n")
            f.write("="*120 + "\n")
            
            # 计算正中间的树索引
            total_trees = len(tree_dump)
            middle_tree_index = total_trees // 2
            
            # 提取正中间的树
            if total_trees > 0:
                f.write(f"\n{'='*60} 正中间树（第{middle_tree_index + 1}棵，共{total_trees}棵） {'='*60}\n")
                middle_tree = tree_dump[middle_tree_index]
                f.write(middle_tree)
                f.write("\n")
                print(f"  提取第{middle_tree_index + 1}棵树的规则，规则长度: {len(middle_tree)} 字符")
            else:
                f.write("\n没有可用的树\n")
                print("  没有可用的树")
    
    print(f"  已保存到: {rule_file}")

# 11. 可视化性能对比
print("\n生成性能可视化图表...")
plt.figure(figsize=(14, 8))
x_pos = np.arange(len(performance_df))
width = 0.25

# 三个柱状图：训练准确率、测试准确率、性能指标
plt.bar(x_pos - width, performance_df['Train_Accuracy'], width, label='训练准确率', alpha=0.8, color='skyblue')
plt.bar(x_pos, performance_df['Test_Accuracy'], width, label='测试准确率', alpha=0.8, color='lightcoral')
plt.bar(x_pos + width, performance_df['Performance'], width, label='性能指标(训练+测试)/2', alpha=0.8, color='lightgreen')

plt.xlabel('模型', fontsize=12)
plt.ylabel('准确率/性能', fontsize=12)
plt.title('决策树模型性能对比（按(训练+测试)/2排名）', fontsize=16, fontweight='bold')
plt.xticks(x_pos, performance_df['Model'], rotation=45)
plt.ylim([0, 1.05])
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()

performance_chart_path = os.path.join(decision_tree_dir, '模型性能对比.png')
plt.savefig(performance_chart_path, dpi=300)
plt.show()

# 12. 保存特征名称
feature_names_path = os.path.join(decision_tree_dir, '特征名称列表.txt')
with open(feature_names_path, 'w', encoding='utf-8') as f:
    f.write(f"决策树使用的特征名称列表（共{len(feature_names)}个特征，排除聚类指标）:\n")
    f.write("="*80 + "\n")
    for i, feature in enumerate(feature_names, 1):
        f.write(f"{i:3d}. {feature}\n")
print(f"特征名称列表已保存到: {feature_names_path}")

# 13. 生成聚类标签描述
print("\n生成聚类标签商业描述...")
cluster_summary = []
for cluster_id in sorted(df['cluster_label'].unique()):
    cluster_data = df[df['cluster_label'] == cluster_id]
    cluster_summary.append({
        'cluster_id': cluster_id,
        '样本数': len(cluster_data),
        '占比': f"{len(cluster_data)/len(df)*100:.1f}%",
        '销量均值': cluster_data['copies_sold'].mean() if 'copies_sold' in df.columns else None,
        '好评率均值': cluster_data['positive_review_rate'].mean() if 'positive_review_rate' in df.columns else None
    })

cluster_summary_df = pd.DataFrame(cluster_summary)
cluster_summary_file = os.path.join(decision_tree_dir, '聚类标签商业描述.csv')
cluster_summary_df.to_csv(cluster_summary_file, index=False, encoding='utf-8')
print(f"聚类标签商业描述已保存到: {cluster_summary_file}")

print(f"\n" + "="*80)
print("决策树分类分析完成!")
print(f"所有结果已保存到: {decision_tree_dir}")
print("="*80)