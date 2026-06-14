import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

# ── 中文字體設定 ───────────────────────────────────────────────────────────
chinese_fonts = ['Microsoft JhengHei', 'Microsoft YaHei', 'SimHei',
                 'PingFang TC', 'Heiti TC', 'Noto Sans CJK TC']
available = [f.name for f in fm.fontManager.ttflist]
chosen = next((f for f in chinese_fonts if f in available), None)
if chosen:
    plt.rcParams['font.family'] = chosen
plt.rcParams['axes.unicode_minus'] = False

# ── 載入資料 ───────────────────────────────────────────────────────────────
df = pd.read_csv('bank-full.csv', sep=';')
df['y'] = df['y'].map({'yes': 1, 'no': 0})

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = le.fit_transform(df[col])

# ── 計算與 y 的相關係數，排除 y 本身 ──────────────────────────────────────
corr = df.corr()['y'].drop('y').sort_values()

# 欄位中文對照
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

# ── 顏色：正相關藍、負相關紅 ──────────────────────────────────────────────
colors = ['#E74C3C' if v < 0 else '#2E86C1' for v in corr]

fig, ax = plt.subplots(figsize=(9, 8))
bars = ax.barh(corr.index, corr.values, color=colors, edgecolor='white', height=0.6)

# 數值標籤
for bar, val in zip(bars, corr.values):
    offset = 0.003 if val >= 0 else -0.003
    ha = 'left' if val >= 0 else 'right'
    ax.text(val + offset, bar.get_y() + bar.get_height()/2,
            f'{val:+.2f}', va='center', ha=ha, fontsize=9)

# 零線
ax.axvline(0, color='black', linewidth=0.8)



# 圖例
from matplotlib.patches import Patch
legend = [Patch(fc='#2E86C1', label='正相關（越高越容易訂閱）'),
          Patch(fc='#E74C3C', label='負相關（越高越不容易訂閱）')]
ax.legend(handles=legend, loc='lower right', fontsize=9)

ax.set_xlabel('與訂閱結果 (y) 的相關係數', fontsize=12)
ax.set_title('各特徵與定存訂閱結果的相關係數', fontsize=14, fontweight='bold')
ax.set_xlim(-0.25, 0.55)

plt.tight_layout()
plt.savefig('correlation_with_y.png', dpi=150)
plt.close()
print("→ correlation_with_y.png 已儲存")