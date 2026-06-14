import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, roc_curve, auc
)

# ══════════════════════════════════════════════════════════════════════════════
# 0. 中文字體設定
# ══════════════════════════════════════════════════════════════════════════════
chinese_fonts = ['Microsoft JhengHei', 'Microsoft YaHei', 'SimHei',
                 'PingFang TC', 'Heiti TC', 'Noto Sans CJK TC']
available = [f.name for f in fm.fontManager.ttflist]
chosen = next((f for f in chinese_fonts if f in available), None)
if chosen:
    plt.rcParams['font.family'] = chosen
plt.rcParams['axes.unicode_minus'] = False

# ══════════════════════════════════════════════════════════════════════════════
# 1. 載入資料
# ══════════════════════════════════════════════════════════════════════════════
print("Loading data...")
df_raw = pd.read_csv('bank-full.csv', sep=';')
df     = pd.read_csv('bank-full.csv', sep=';')

df['y'] = df['y'].map({'yes': 1, 'no': 0})
le = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = le.fit_transform(df[col])

# ══════════════════════════════════════════════════════════════════════════════
# 2. 特徵相關性熱力圖
# ══════════════════════════════════════════════════════════════════════════════
print("Plotting correlation matrix...")
plt.figure(figsize=(15, 10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=150)
plt.close()
print("  -> correlation_matrix.png 已儲存")

# ══════════════════════════════════════════════════════════════════════════════
# 3. 各特徵與 y 的相關係數長條圖
# ══════════════════════════════════════════════════════════════════════════════
print("Plotting correlation with y...")
corr = df.corr()['y'].drop('y').sort_values()

label_map = {
    'duration' : 'duration\n通話時長',
    'pdays'    : 'pdays\n距上次聯繫天數',
    'previous' : 'previous\n先前聯繫次數',
    'balance'  : 'balance\n帳戶餘額',
    'age'      : 'age\n年齡',
    'day'      : 'day\n聯繫日期',
    'month'    : 'month\n聯繫月份',
    'campaign' : 'campaign\n本次聯繫次數',
    'job'      : 'job\n職業',
    'marital'  : 'marital\n婚姻狀況',
    'education': 'education\n教育程度',
    'default'  : 'default\n信用違約',
    'housing'  : 'housing\n房屋貸款',
    'loan'     : 'loan\n個人貸款',
    'contact'  : 'contact\n聯繫方式',
    'poutcome' : 'poutcome\n前次活動結果',
}
corr.index = [label_map.get(i, i) for i in corr.index]
colors = ['#E74C3C' if v < 0 else '#2E86C1' for v in corr]

fig, ax = plt.subplots(figsize=(9, 8))
bars = ax.barh(corr.index, corr.values, color=colors, edgecolor='white', height=0.6)
for bar, val in zip(bars, corr.values):
    offset = 0.003 if val >= 0 else -0.003
    ha = 'left' if val >= 0 else 'right'
    ax.text(val + offset, bar.get_y() + bar.get_height()/2,
            f'{val:+.2f}', va='center', ha=ha, fontsize=9)
ax.axvline(0, color='black', linewidth=0.8)


ax.set_xlabel('與訂閱結果 (y) 的相關係數', fontsize=12)
ax.set_title('各特徵與定存訂閱結果的相關係數', fontsize=14, fontweight='bold')
ax.set_xlim(-0.25, 0.55)
plt.tight_layout()
plt.savefig('correlation_with_y.png', dpi=150)
plt.close()
print("  -> correlation_with_y.png 已儲存")

# ══════════════════════════════════════════════════════════════════════════════
# 4. 通話時長 vs 訂閱 Boxplot
# ══════════════════════════════════════════════════════════════════════════════
print("Plotting duration boxplot...")
plt.figure(figsize=(10, 6))
sns.boxplot(x='y', y='duration', data=df_raw)
plt.title('Call Duration vs Subscription')
plt.tight_layout()
plt.savefig('duration_vs_subscription.png', dpi=150)
plt.close()
print("  -> duration_vs_subscription.png 已儲存")

# ══════════════════════════════════════════════════════════════════════════════
# 5. 通話時長 KDE 分布（中位數 + 平均數 + 右偏說明）
# ══════════════════════════════════════════════════════════════════════════════
print("Plotting duration KDE...")
no_sub  = df_raw[df_raw['y'] == 'no']['duration']
yes_sub = df_raw[df_raw['y'] == 'yes']['duration']

fig, ax = plt.subplots(figsize=(10, 6))
sns.kdeplot(no_sub,  ax=ax, fill=True, color='#5B9BD5', alpha=0.4,
            label='未訂閱 (y=0)', linewidth=2)
sns.kdeplot(yes_sub, ax=ax, fill=True, color='#E74C3C', alpha=0.4,
            label='已訂閱 (y=1)', linewidth=2)

med_no  = no_sub.median();  avg_no  = no_sub.mean()
med_yes = yes_sub.median(); avg_yes = yes_sub.mean()

ax.axvline(med_no,  color='#2471A3', linestyle='--', linewidth=1.5,
           label=f'未訂閱 中位數：{med_no:.0f} 秒')
ax.axvline(avg_no,  color='#2471A3', linestyle=':',  linewidth=1.8,
           label=f'未訂閱 平均數：{avg_no:.0f} 秒')
ax.axvline(med_yes, color='#C0392B', linestyle='--', linewidth=1.5,
           label=f'已訂閱 中位數：{med_yes:.0f} 秒')
ax.axvline(avg_yes, color='#C0392B', linestyle=':',  linewidth=1.8,
           label=f'已訂閱 平均數：{avg_yes:.0f} 秒')

y_arrow_no  = 0.0012
y_arrow_yes = 0.0005
ax.annotate('', xy=(avg_no, y_arrow_no), xytext=(med_no, y_arrow_no),
            arrowprops=dict(arrowstyle='->', color='#2471A3', lw=1.5))
ax.text((med_no+avg_no)/2, y_arrow_no+0.00008,
        f'+{avg_no-med_no:.0f}秒', ha='center', fontsize=9, color='#2471A3')
ax.annotate('', xy=(avg_yes, y_arrow_yes), xytext=(med_yes, y_arrow_yes),
            arrowprops=dict(arrowstyle='->', color='#C0392B', lw=1.5))
ax.text((med_yes+avg_yes)/2, y_arrow_yes+0.00008,
        f'+{avg_yes-med_yes:.0f}秒', ha='center', fontsize=9, color='#C0392B')


ax.set_xlim(0, 3000)
ax.set_xlabel('通話時長（秒）', fontsize=12)
ax.set_ylabel('機率密度', fontsize=12)
ax.set_title('訂閱 vs 未訂閱客戶的通話時長分布', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax2 = ax.secondary_xaxis('top', functions=(lambda x: x/60, lambda x: x*60))
ax2.set_xlabel('通話時長（分鐘）', fontsize=11)
ax2.set_xticks([0, 1, 3, 5, 10, 15, 20, 30, 50])
plt.tight_layout()
plt.savefig('duration_distribution_kde.png', dpi=150)
plt.close()
print("  -> duration_distribution_kde.png 已儲存")

# ══════════════════════════════════════════════════════════════════════════════
# 6. 建立隨機森林模型
# ══════════════════════════════════════════════════════════════════════════════
print("\nBuilding Random Forest model...")
X = df.drop('y', axis=1)
y = df['y']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# ══════════════════════════════════════════════════════════════════════════════
# 7. 特徵重要性
# ══════════════════════════════════════════════════════════════════════════════
print("Plotting feature importance...")
fi_df = (pd.DataFrame({'Feature': X.columns,
                        'Importance': model.feature_importances_})
           .sort_values('Importance', ascending=False))
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=fi_df)
plt.title('Feature Importance for Subscribing Term Deposit')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150)
plt.close()
print("  -> feature_importance.png 已儲存")

# ══════════════════════════════════════════════════════════════════════════════
# 8. 混淆矩陣（視覺化 + TP/TN/FP/FN 標註）
# ══════════════════════════════════════════════════════════════════════════════
print("Plotting confusion matrix...")
cm     = confusion_matrix(y_test, y_pred)
labels = ['No (未訂閱)', 'Yes (已訂閱)']
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels,
            linewidths=0.5, ax=ax)
ax.set_xlabel('預測值', fontsize=12)
ax.set_ylabel('實際值', fontsize=12)
ax.set_title('混淆矩陣 (Confusion Matrix)', fontsize=14, fontweight='bold')
for i in range(2):
    for j in range(2):
        tag = {(0,0):'TN',(0,1):'FP',(1,0):'FN',(1,1):'TP'}[(i,j)]
        ax.text(j+0.5, i+0.75, tag, ha='center', va='center',
                fontsize=9, color='gray')
plt.tight_layout()
plt.savefig('confusion_matrix_visual.png', dpi=150)
plt.close()
print("  -> confusion_matrix_visual.png 已儲存")

# ══════════════════════════════════════════════════════════════════════════════
# 9. ROC 曲線 + AUC
# ══════════════════════════════════════════════════════════════════════════════
print("Plotting ROC curve...")
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc     = auc(fpr, tpr)
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr, tpr, color='steelblue', lw=2,
        label=f'ROC 曲線 (AUC = {roc_auc:.3f})')
ax.plot([0,1],[0,1], color='gray', lw=1, linestyle='--', label='隨機猜測基準線')
ax.fill_between(fpr, tpr, alpha=0.08, color='steelblue')
ax.set_xlabel('偽陽性率 (False Positive Rate)', fontsize=12)
ax.set_ylabel('真陽性率 (True Positive Rate)', fontsize=12)
ax.set_title('ROC 曲線', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=11)
ax.set_xlim([0,1]); ax.set_ylim([0,1.02])
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=150)
plt.close()
print("  -> roc_curve.png 已儲存")

# ══════════════════════════════════════════════════════════════════════════════
# 10. 各職業訂閱率
# ══════════════════════════════════════════════════════════════════════════════
print("Plotting subscription by job...")
job_rate = (df_raw.groupby('job')['y']
            .apply(lambda x: (x=='yes').mean()*100)
            .sort_values(ascending=False).reset_index())
job_rate.columns = ['職業', '訂閱率(%)']
fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.barh(job_rate['職業'], job_rate['訂閱率(%)'],
               color=sns.color_palette('Blues_r', len(job_rate)))
ax.bar_label(bars, fmt='%.1f%%', padding=3, fontsize=9)
ax.set_xlabel('訂閱率 (%)', fontsize=12)
ax.set_title('各職業別定存訂閱率', fontsize=14, fontweight='bold')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('subscription_by_job.png', dpi=150)
plt.close()
print("  -> subscription_by_job.png 已儲存")

# ══════════════════════════════════════════════════════════════════════════════
# 11. 各月份訂閱率
# ══════════════════════════════════════════════════════════════════════════════
print("Plotting subscription by month...")
month_order = ['jan','feb','mar','apr','may','jun',
               'jul','aug','sep','oct','nov','dec']
month_rate = (df_raw.groupby('month')['y']
              .apply(lambda x: (x=='yes').mean()*100)
              .reindex(month_order).dropna().reset_index())
month_rate.columns = ['月份', '訂閱率(%)']
month_rate['月份'] = month_rate['月份'].str.upper()
fig, ax = plt.subplots(figsize=(10, 5))
colors = ['#E74C3C' if v == month_rate['訂閱率(%)'].max()
          else '#5B9BD5' for v in month_rate['訂閱率(%)']]
bars = ax.bar(month_rate['月份'], month_rate['訂閱率(%)'], color=colors)
ax.bar_label(bars, fmt='%.1f%%', padding=3, fontsize=9)
ax.set_xlabel('月份', fontsize=12)
ax.set_ylabel('訂閱率 (%)', fontsize=12)
ax.set_title('各月份定存訂閱率（紅色 = 最高）', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('subscription_by_month.png', dpi=150)
plt.close()
print("  -> subscription_by_month.png 已儲存")

# ══════════════════════════════════════════════════════════════════════════════
# 12. 通話時長區間訂閱率
# ══════════════════════════════════════════════════════════════════════════════
print("Plotting subscription by duration...")
df_raw['duration_bin'] = pd.cut(
    df_raw['duration'],
    bins=[0, 60, 180, 300, 600, 900, 99999],
    labels=['<1分', '1-3分', '3-5分', '5-10分', '10-15分', '>15分'])
dur_rate = (df_raw.groupby('duration_bin', observed=True)['y']
            .apply(lambda x: (x=='yes').mean()*100).reset_index())
dur_rate.columns = ['通話時長', '訂閱率(%)']
fig, ax = plt.subplots(figsize=(8, 5))
colors = sns.color_palette('RdYlGn', len(dur_rate))
bars = ax.bar(dur_rate['通話時長'].astype(str), dur_rate['訂閱率(%)'], color=colors)
ax.bar_label(bars, fmt='%.1f%%', padding=3, fontsize=10)
ax.set_xlabel('通話時長', fontsize=12)
ax.set_ylabel('訂閱率 (%)', fontsize=12)
ax.set_title('通話時長與訂閱率關係（行銷建議）', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('subscription_by_duration.png', dpi=150)
plt.close()
print("  -> subscription_by_duration.png 已儲存")

# ══════════════════════════════════════════════════════════════════════════════
# 13. 模型指標摘要
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("模型評估摘要")
print("="*55)
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"AUC      : {roc_auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['未訂閱', '已訂閱']))
print("="*55)
