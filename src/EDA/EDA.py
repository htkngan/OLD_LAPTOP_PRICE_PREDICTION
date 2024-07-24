# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import warnings
from scipy import stats
from scipy.stats import f_oneway
from sklearn.impute import KNNImputer
from statsmodels.graphics.factorplots import interaction_plot
from sklearn.preprocessing import LabelEncoder
import statsmodels.api as sm
from statsmodels.formula.api import ols
warnings.filterwarnings("ignore")


# %%
df = pd.read_csv('D:\HK5\DS105\Đồ án\src\data\data_preprocess_step2.csv')

# lấy các cột chỉ có 2 giá trị 1 và 0
binary_cols = list(df.columns[(df.nunique() == 2) & (df.isin([0, 1]).all())])

# lấy các biến số và biến phân loại
quanti_cols = df.select_dtypes(include='number').columns
quali_df = df.select_dtypes(include='object')
quali_cols = list(quali_df.columns)

quanti_cols = [col for col in quanti_cols if col not in binary_cols]

#Kiểm định độ ảnh hưởng đến biến mục tiêu của các biến có tỉ lệ khuyết cao
# %%
# Kiểm tra độ tương quan của thuộc tính 'battery_capacity'
df_cleaned_2 = df.dropna(subset=['battery_capacity'] + quanti_cols + binary_cols)

# Calculate correlation
correlation, p_value = stats.pearsonr(df_cleaned_2['battery_capacity'], df_cleaned_2['price'])
print(f"Correlation: {correlation}")
print(f"P-value: {p_value}")

'''
Kết quả:
Correlation: 0.5728233639207811
P-value: 3.4146741615613295e-21
Kết luận: không thể bỏ thuộc tính 'battery_capacity'
'''

# Xem ảnh hưởng của cột 'cell_num'
df_cleaned_1 = df.dropna(subset=['cell_num']+ quanti_cols + binary_cols)
# %%
#boxplot biểu diễn phân phối
sns.boxplot(x='cell_num', y='price', data=df_cleaned_1)
plt.show()

# Perform ANOVA to get p-value
category_groups = [df_cleaned_1['price'][df_cleaned_1['cell_num'] == category] for category in df_cleaned_1['cell_num'].unique()]

f_statistic, p_value = f_oneway(*category_groups)

# Print the p-value
print("P-value:", p_value)

'''
Kết quả:
P-value: 1.5049699538979057e-19
Kết luận: không thể bỏ thuộc tính 'cell_num'
'''

# Xem ảnh hưởng của cột 'screen_tech'
df_cleaned_3 = df.dropna(subset=['screen_tech', 'price'])


#boxplot biểu diễn phân phối
ax = sns.boxplot(x='screen_tech', y='price', data=df_cleaned_3)
__ = ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
plt.show()

# Perform ANOVA to get p-value
category_groups = [df_cleaned_3['price'][df_cleaned_3['screen_tech'] == category] for category in df_cleaned_3['screen_tech'].unique()]

f_statistic, p_value = f_oneway(*category_groups)

# Print the p-value
print("P-value:", p_value)

'''
Kết quả:
P-value: 1.028722014465598e-06
Kết luận: không thể bỏ thuộc tính 'screen_tech'
'''
# %%
#Sử dụng pearson và anova để chọn ra các biến phù hợp để làm đầu vào cho model
#Điều kiện: p-value < 0.05 (coef > 0.3)

selected_cols = []

# Biến liên tục
for col in quanti_cols:
    if col == 'price':
        continue

    correlation, p_value = stats.pearsonr(df['price'], df[col])
    print(col)
    print(f"Correlation: {correlation}")
    print(f"P-value: {p_value}")
    print('\n')

    if p_value < 0.05 and round(abs(correlation), 1) >= 0.3:
        selected_cols.append(col)

# Biến phân loại
for col in quali_cols:
    category_groups = [df['price'][df[col] == category] for category in df[col].unique()]

    f_statistic, p_value = f_oneway(*category_groups)

    print(col, "P-value:", p_value, '\n')

    if p_value < 0.05:
        selected_cols.append(col)

# %%
#Plot đồ thị phân tích đơn biến
#Biến phân loại
fig, ax = plt.subplots(nrows=43)

# Vẽ barplot cho các biến phân loại
for column in quali_cols + binary_cols:
    plt.figure()
    factor = df[column].nunique()
    plt.figure(figsize=(2 + factor * 0.4, 6))
    temp = df[column].value_counts().sort_values(ascending=False)
    ax = sns.barplot(x=temp.index, y=temp, order=temp.index)

    # Size của x y ticks
    plt.xticks(fontsize=12)
    plt.ylabel("count")
    plt.yticks(fontsize=12)
    # Size của x y labels
    axes = plt.gca()
    axes.xaxis.label.set_size(12)
    axes.yaxis.label.set_size(12)

    # Bỏ viền xung quanh
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Chú thích cho từng cột
    total = len(df[column])
    for p in ax.patches:
        percentage = '{:.1f}%'.format(100 * p.get_height() / total)
        x = p.get_x() + p.get_width() / 2 - 0.33
        y = p.get_y() + p.get_height() + 0.5
        _ = ax.annotate(percentage, (x, y), size=12)

    # Xoay x ticks
    __ = ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

    # Tiêu đề
    plt.title(f'{column}', fontsize=13.5)

    plt.show()
# %%
#Boxplot cho biến phân loại
cate_p_value = pd.DataFrame({'feature': [], 'p_value': []})

for column in quali_cols + binary_cols:
    plt.figure()
    factor = df[column].nunique()
    plt.figure(figsize=(1.5 + factor * 1, 10))


    # Tạo group rồi sort theo median trước khi vẽ
    grouped = df[[column, 'price']].groupby([column])
    df2 = pd.DataFrame({col: vals['price'] for col, vals in grouped})
    if str(df2.columns[0]).replace('.', '').isnumeric():
        df2 = df2[df2.columns.sort_values()]
    else:
        meds = df2.median()
        meds.sort_values(ascending=True, inplace=True)
        df2 = df2[meds.index]
    ax = sns.boxplot(data=df2)

    # Bỏ viền xung quanh
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Size của x y ticks
    plt.xticks(fontsize=15.65)
    plt.yticks(fontsize=15.65)
    # Size của x y labels
    axes = plt.gca()
    axes.xaxis.label.set_size(15.65)
    axes.yaxis.label.set_size(15.65)

    # Kích thước mỗi bar
    #change_width(ax, 0.7)

    # Vẽ ticks của trục y
    __ = ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    # Custom ticks của trục y
    labels = [str(int(item / 1e6)) + ' tr' for item in ax.get_yticks()]
    __ = ax.set_yticklabels(labels)

    # Tính ANOVA nhằm kiểm tra xem có sự khác nhau giữa các nhóm hay không
    model = ols(f'price ~ {column}', data=df).fit()
    aov = sm.stats.anova_lm(model, typ=2)
    p_value = aov['PR(>F)'][f'{column}']
    cate_p_value = cate_p_value.append({'feature': column, 'p_value': p_value}, ignore_index=True)

    # Tiêu đề
    ax = plt.title(f'price vs {column}\n(p_value: {p_value:.2e})', fontsize=15.65)

    plt.tight_layout()
    plt.show()
    # break
    print('\n')
# %%
#Regplot cho biến liên tục
for column in ['battery_capacity', 'ppi']:
    ax = sns.regplot(x = column, y = 'price', data = df, color = '#e34a33')

    # Bỏ viền xung quanh
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    plt.show()


# %%
#Plot đồ thị biểu diễn phân phối của các thuộc tính - đa số thuộc phân phối chuẩn
for column in quanti_cols:
    # if column == 'used_price':
    #     continue
    plt.figure(figsize=(6, 6))
    ax = sns.kdeplot(x=df[column], color = '#e34a33')

    # Bỏ viền xung quanh
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    __ = plt.xlabel(column)

    # Size của x y ticks
    plt.xticks(fontsize=15.65)
    plt.yticks(fontsize=15.65)
    # Size của x y labels
    axes = plt.gca()
    axes.xaxis.label.set_size(15.65)
    axes.yaxis.label.set_size(15.65)

    # Tiêu đề
    plt.title(f'{column}', fontsize=15.65)

    plt.tight_layout()
    plt.show()
    plt.clf()
# %%
#Interaction plot dành cho đa biến
p_values = {}
for i in range(len(quali_cols)):
    for j in range(len(quali_cols)):
        num_groups = df.groupby([quali_cols[i], quali_cols[j]]).ngroups
        if i == j or num_groups < 5 :
            continue

        if df[quali_cols[i]].isnull().any() or df[quali_cols[j]].isnull().any():
            continue

        # convert all values to string to fix x_tick not recognize
        df_temp = quali_df.dropna(subset=[quali_cols[i], quali_cols[j]])

        df_temp[quali_cols[i]] = df_temp[quali_cols[i]].apply(lambda x: str(x))

        # Tính p-value cho tương tác của 2 biến. p < 0.05 => có ý nghĩa thống kê => có tương tác giữa 2 biến
        model = ols(f'price ~ {quali_cols[i]} + {quali_cols[j]} + {quali_cols[i]}:{quali_cols[j]}',
                    data=df).fit()
        aov = sm.stats.anova_lm(model, typ=2)
        p_value = aov['PR(>F)'][f'{quali_cols[i]}:{quali_cols[j]}']

        if p_value > 0.05:
            continue

        plt.figure(figsize=(10,10))
        # Vẽ interaction plot
        interaction_plot(x=df_temp[quali_cols[i]], trace=df_temp[quali_cols[j]], response=df['price'],
                         plottype='both', ms=17)

        # Size của plot

        fig = plt.gcf()
        fig.set_size_inches(5, 7)

        # Size của legend
        plt.legend(fontsize=10)

        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        axes = plt.gca()

        # Size của x y labels
        axes.xaxis.label.set_size(10)
        axes.yaxis.label.set_size(10)

        # Bỏ viền xung quanh
        ax = plt.subplot(111)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        # Vẽ ticks của trục x
        __ = ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        labels = [str(int(item / 1e6)) + ' tr' for item in ax.get_yticks()]
        # Custom ticks của trục y
        __ = ax.set_yticklabels(labels)

        ___ = plt.title(f'price vs {quali_cols[i]}:{quali_cols[j]}\n(p_value: {p_value:.2e})',
                        fontdict={'size': 12})
        p_values[f'{quali_cols[i]}|{quali_cols[j]}'] = p_value
        plt.tight_layout()
        plt.show()
        # break
    # break


# %%
#lmplot biểu diễn sự tương quan đặc biệt của biến liên tục và biến phân loại
sns.lmplot(data=df, x='battery_capacity', y='price', hue='cell_num')
plt.show()

# %%
