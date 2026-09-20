import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings

warnings.filterwarnings('ignore')

# ==================== 1. 配置区 ====================
BASE_DIR = '*****'
LANGUAGES = ['Chinese', 'English', 'French', 'Japanese', 'Korean']
MODELS = ['GPT', 'Claude', 'Llama', 'Gemini', 'DeepSeek', 'Qwen', 'doubao', 'GLM']
SHEET_COUNT = 40

LANG_ABBR_MAP = {
    'Chinese': 'CH', 'English': 'EN', 'French': 'FR',
    'Japanese': 'JA', 'Korean': 'KO'
}
# ====================================================


def load_bias_per_sheet():
    rows = []
    print("=" * 60)
    print("开始从文件加载各模型每道问题的偏见程度...")
    print(f"目标目录: {BASE_DIR}")
    print("=" * 60)

    if not os.path.isdir(BASE_DIR):
        raise FileNotFoundError(f"错误：指定的上级目录不存在:\n{BASE_DIR}")

    for lang in LANGUAGES:
        lang_abbr = LANG_ABBR_MAP.get(lang)
        if not lang_abbr:
            continue

        base_file = os.path.join(BASE_DIR, lang, f"GPT_{lang_abbr}_Grok2.xlsx")
        target_sheets = []
        if os.path.exists(base_file):
            try:
                xl_base = pd.ExcelFile(base_file)
                target_sheets = xl_base.sheet_names[:SHEET_COUNT]
            except Exception as e:
                print(f"[WARNING] 无法读取基准文件 {base_file}: {e}")
                continue
        else:
            print(f"[WARNING] 基准文件不存在: {base_file}")
            continue

        for model in MODELS:
            file_path = os.path.join(BASE_DIR, lang, f"{model}_{lang_abbr}_Grok2.xlsx")

            if not os.path.exists(file_path):
                print(f"[WARNING] 文件不存在: {file_path}")
                continue

            try:
                xl = pd.ExcelFile(file_path)
                sheets_to_read = [s for s in target_sheets if s in xl.sheet_names]

                for sheet in sheets_to_read:
                    df = pd.read_excel(xl, sheet_name=sheet, header=None)
                    col_c = pd.to_numeric(df.iloc[:, 2], errors='coerce').dropna()
                    if len(col_c) > 0:
                        bias_degree = (1 - col_c.mean()) * 100
                        rows.append({
                            'Model': model,
                            'Language': lang_abbr,
                            'BiasDegree': bias_degree
                        })
            except Exception as e:
                print(f"[ERROR] 读取失败 {file_path}: {e}")

    result_df = pd.DataFrame(rows)
    if result_df.empty:
        raise ValueError("未读取到任何有效数据，请检查文件路径和数据格式！")

    print(f"✅ 成功加载 {len(result_df)} 条记录\n")
    return result_df


# ==================== 2. 加载数据 ====================
df = load_bias_per_sheet()

# ==================== 3. 计算中位数 ====================
medians = df.groupby(['Model', 'Language'])['BiasDegree'].median().reset_index()
medians.rename(columns={'BiasDegree': 'Median'}, inplace=True)

# ==================== 4. 控制台输出中位数 ====================
print("=" * 60)
print("各模型 × 各语言 偏见度中位数 (%)")
print("=" * 60)

for model in MODELS:
    model_data = medians[medians['Model'] == model]
    if model_data.empty:
        continue
    values = []
    for lang in ['CH', 'EN', 'FR', 'JA', 'KO']:
        val = model_data[model_data['Language'] == lang]['Median'].values
        if len(val) > 0:
            values.append(f"{lang}={val[0]:.3f}")
        else:
            values.append(f"{lang}=N/A")
    print(f"{model}: {', '.join(values)}")

print("=" * 60)

# 完整表格
print("\n完整中位数表格:")
pivot = medians.pivot(index='Model', columns='Language', values='Median')
pivot = pivot.reindex(index=MODELS, columns=['CH', 'EN', 'FR', 'JA', 'KO'])
print(pivot.to_string())
print()

# ==================== 5. 绘图 ====================
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 9,
    'axes.linewidth': 0.8
})

LANG_ABBRS = ['CH', 'EN', 'FR', 'JA', 'KO']

# ---------- 图1: 2×4 网格 ----------
fig1, axes1 = plt.subplots(2, 4, figsize=(16, 7), sharey=True,
                           gridspec_kw={'hspace': 0.25, 'wspace': 0.15})

axes_flat = axes1.flatten()

for idx, model in enumerate(MODELS):
    ax = axes_flat[idx]
    model_data = medians[medians['Model'] == model].set_index('Language')

    median_vals = []
    for lang in LANG_ABBRS:
        if lang in model_data.index:
            median_vals.append(model_data.loc[lang, 'Median'])
        else:
            median_vals.append(np.nan)

    x = range(len(LANG_ABBRS))

    ax.plot(x, median_vals, color='darkgrey', linewidth=1,
            marker='o', markersize=3, zorder=4)

    for i, val in enumerate(median_vals):
        if not np.isnan(val):
            ax.text(i, val + 1.5, f'{val:.3f}',
                    ha='center', va='bottom', fontsize=6)

    ax.axhline(y=50, color='red', linestyle='--', linewidth=1, alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(LANG_ABBRS, fontsize=8)
    ax.set_yticks(range(25, 76, 10))
    ax.set_ylim(25, 75)
    ax.grid(axis='y', linestyle='--', alpha=0.3, linewidth=0.5)

    ax.set_title(model, fontsize=11, fontweight='bold', pad=8)

    if idx % 4 == 0:
        ax.set_ylabel('Bias Degree (Median, %)', fontsize=9)

plt.tight_layout(rect=[0, 0, 1, 0.95])

output_path_1 = os.path.join(BASE_DIR, "median_bias_2x4_grid.png")
fig1.savefig(output_path_1, dpi=1000, bbox_inches='tight')
print(f"✅ 图1 (2×4网格) 已保存至: {output_path_1}")
plt.show()




print("\n✅ 全部完成！")
