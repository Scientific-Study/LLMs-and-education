import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# ==================== 1. 配置区 ====================
BASE_DIR = '****'
LANGUAGES = ['Chinese', 'English', 'French', 'Japanese', 'Korean']
MODELS = ['GPT', 'Claude', 'Llama', 'Gemini', 'DeepSeek', 'Qwen', 'doubao', 'GLM']
SHEET_COUNT = 40

LANG_ABBR_MAP = {
    'Chinese': 'CH', 'English': 'EN', 'French': 'FR',
    'Japanese': 'JA', 'Korean': 'KO'
}
# ====================================================


def calculate_global_weighted_bias():
    """
    计算每个模型在每种语言下的全局加权平均偏见率(%)
    公式: (1 - 所有Sheet所有有效C列值的全局均值) * 100
    返回: dict[model][lang_abbr] = float | np.nan
    """
    result = {model: {} for model in MODELS}

    print("=" * 60)
    print("开始计算各模型全局加权平均偏见率（与代码1逻辑一致）...")
    print(f"目标目录: {BASE_DIR}")
    print("=" * 60)

    if not os.path.isdir(BASE_DIR):
        raise FileNotFoundError(f"错误：指定的上级目录不存在:\n{BASE_DIR}")

    for lang in LANGUAGES:
        lang_abbr = LANG_ABBR_MAP.get(lang)
        if not lang_abbr:
            continue

        for model in MODELS:
            file_path = os.path.join(BASE_DIR, lang, f"{model}_{lang_abbr}_Grok2.xlsx")
            all_scores = []

            if not os.path.exists(file_path):
                print(f"[WARNING] 文件不存在: {file_path}")
                result[model][lang_abbr] = np.nan
                continue

            try:
                xl = pd.ExcelFile(file_path)
                sheets = xl.sheet_names[:SHEET_COUNT]
                for sheet in sheets:
                    df = pd.read_excel(xl, sheet_name=sheet, header=None)
                    col_c = pd.to_numeric(df.iloc[:, 2], errors='coerce').dropna()
                    all_scores.extend(col_c.tolist())
            except Exception as e:
                print(f"[ERROR] 读取失败 {file_path}: {e}")
                result[model][lang_abbr] = np.nan
                continue

            if len(all_scores) > 0:
                bias = round((1 - sum(all_scores) / len(all_scores)) * 100, 2)
                result[model][lang_abbr] = bias
                print(f"[INFO] {model:>10s} | {lang:>8s} | 样本数: {len(all_scores):>6d} | 偏见率: {bias:.2f}%")
            else:
                print(f"[WARNING] {model:>10s} | {lang:>8s} | 无有效数据")
                result[model][lang_abbr] = np.nan

    print("✅ 全局加权偏见率计算完成！\n")
    return result


# ==================== 2. 执行数据计算 ====================
global_weighted_bias = calculate_global_weighted_bias()

# ==================== 3. 控制台打印验证表 ====================
print("=" * 80)
print("各模型在不同语言下的全局加权偏见程度（Bias Degree %）")
print("=" * 80)

header = f"{'Model':>12s}"
for lang in LANGUAGES:
    header += f" | {lang:>10s}"
print(header)
print("-" * 80)

for model in MODELS:
    row = f"{model:>12s}"
    for lang in LANGUAGES:
        lang_abbr = LANG_ABBR_MAP[lang]
        val = global_weighted_bias[model].get(lang_abbr, np.nan)
        if not np.isnan(val):
            row += f" | {val:>9.2f}%"
        else:
            row += f" | {'N/A':>10s}"
    print(row)

print("=" * 80)

# ==================== 4. 绘图（折线图，Y轴25-75） ====================
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 8

models = MODELS
languages = ['CH', 'EN', 'FR', 'JA', 'KO']

fig, axes = plt.subplots(1, 5, figsize=(12, 4), sharey=True, gridspec_kw={'wspace': 0.05})

for i, lang in enumerate(languages):
    ax = axes[i]

    # ✅ 直接使用预计算的全局加权平均值
    means = np.array([global_weighted_bias[m].get(lang, np.nan) for m in models])

    valid_mask = ~np.isnan(means)
    valid_indices = np.where(valid_mask)[0]
    valid_means = means[valid_mask]

    if len(valid_indices) > 0:
        ax.plot(valid_indices, valid_means, color='darkgrey', linewidth=1,
                marker='o', markersize=3, zorder=4)
        for j, mean_val in zip(valid_indices, valid_means):
            ax.text(j, mean_val + 1.5, f'{mean_val:.1f}',
                    ha='center', va='bottom', fontsize=6)

    ax.set_title(lang, fontsize=10, pad=10)
    if i == 0:
        ax.set_ylabel('Bias degree (%)', fontsize=9)

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=8)

    # y=50 红色虚线参考线
    ax.axhline(y=50, color='red', linestyle='--', linewidth=1, alpha=0.7)

    # Y轴刻度25-75，步长10
    ax.set_yticks(range(25, 76, 10))
    ax.set_ylim(25, 75)
    ax.grid(axis='y', linestyle='--', alpha=0.3, linewidth=0.5)

plt.tight_layout(rect=[0, 0.1, 1, 0.93])

output_path = os.path.join(BASE_DIR, "line_trend_only.png")
plt.savefig(output_path, dpi=1000, bbox_inches='tight')
print(f"\n✅ 图表已成功保存至: {output_path}")
plt.show()
