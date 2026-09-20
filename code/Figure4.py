import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings('ignore')

# ==================== 1. 配置区 ====================
BASE_DIR = '*****'
LANGUAGES = ['Chinese', 'English', 'French', 'Japanese', 'Korean']
MODELS = ['GPT', 'Claude', 'Llama', 'Gemini', 'DeepSeek', 'Qwen', 'doubao', 'GLM']
SHEET_COUNT = 40
QUESTION_COUNT = 25

LANG_ABBR_MAP = {
    'Chinese': 'CH', 'English': 'EN', 'French': 'FR',
    'Japanese': 'JA', 'Korean': 'KO'
}

# 偏见维度分组（按问题编号）
BIAS_GROUPS = {
    'Gender stereotypes': [1, 2],
    'Group stereotypes': [3, 4, 5, 6],
    'Racial and ethnic stereotypes': [7, 8, 9, 10],
    'Gender prejudice': [11, 12, 13, 14],
    'Group prejudice': [15, 16, 17, 18, 19],
    'Racial, ethnic and linguistic prejudice': [20, 21, 22, 23, 24, 25]
}
# ====================================================


def calculate_bias_by_group():
    """
    计算每个模型在每种语言下、每个偏见维度的偏见度(%)。

    数据结构：
    - 每个Excel文件有40个Sheet
    - 每个Sheet有26行（第1行是表头，第2~26行是问题1~25的数据）
    - 问题i的评分 = 40个Sheet中第i+1行的C列值（共40个值）

    公式: Bias Degree = (1 - mean(所有相关评分)) * 100
    返回: dict[model][lang_abbr][bias_group] = float | np.nan
    """
    result = {model: {lang_abbr: {} for lang_abbr in LANG_ABBR_MAP.values()} for model in MODELS}

    print("=" * 60)
    print("开始计算各模型各偏见维度的偏见率...")
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

            if not os.path.exists(file_path):
                print(f"[WARNING] 文件不存在: {file_path}")
                for group_name in BIAS_GROUPS:
                    result[model][lang_abbr][group_name] = np.nan
                continue

            try:
                xl = pd.ExcelFile(file_path)
                sheets = xl.sheet_names[:SHEET_COUNT]

                # 读取所有Sheet的数据，构建 问题编号 -> 评分列表 的映射
                # question_scores[q] = 40个Sheet中问题q的C列评分列表
                question_scores = {q: [] for q in range(1, QUESTION_COUNT + 1)}

                for sheet_name in sheets:
                    df = pd.read_excel(xl, sheet_name=sheet_name, header=0)
                    # header=0 表示第1行是表头，数据从第2行开始
                    # 此时 df 的第0行 = 原Sheet的第2行 = 问题1
                    # df 的第1行 = 原Sheet的第3行 = 问题2
                    # ...
                    # df 的第24行 = 原Sheet的第26行 = 问题25

                    for q_idx in range(QUESTION_COUNT):
                        if q_idx < len(df):
                            # C列 = 第3列（索引2）
                            if df.shape[1] > 2:
                                val = pd.to_numeric(df.iloc[q_idx, 2], errors='coerce')
                                if not pd.isna(val):
                                    question_scores[q_idx + 1].append(val)

                # 打印调试信息
                if model == MODELS[0] and lang == LANGUAGES[0]:
                    print(f"\n[DEBUG] 示例文件: {file_path}")
                    print(f"[DEBUG] Sheet数量: {len(sheets)}")
                    print(f"[DEBUG] 问题1的评分数量: {len(question_scores[1])}")
                    print(f"[DEBUG] 问题1的评分示例: {question_scores[1][:5]}...")

                # 对每个偏见维度计算偏见度
                for group_name, qnums in BIAS_GROUPS.items():
                    group_scores = []

                    for qnum in qnums:
                        group_scores.extend(question_scores[qnum])

                    if len(group_scores) > 0:
                        bias = round((1 - sum(group_scores) / len(group_scores)) * 100, 3)
                        result[model][lang_abbr][group_name] = bias
                    else:
                        result[model][lang_abbr][group_name] = np.nan

            except Exception as e:
                print(f"[ERROR] 读取失败 {file_path}: {e}")
                for group_name in BIAS_GROUPS:
                    result[model][lang_abbr][group_name] = np.nan
                continue

            # 打印结果
            bias_str = " | ".join(
                f"{gn[:6]}:{result[model][lang_abbr][gn]:.3f}" if not np.isnan(result[model][lang_abbr][gn]) else f"{gn[:6]}:N/A"
                for gn in BIAS_GROUPS
            )
            print(f"[INFO] {model:>10s} | {lang:>8s} | {bias_str}")

    print("\n✅ 各维度偏见率计算完成！\n")
    return result


# ==================== 2. 执行数据计算 ====================
bias_by_group = calculate_bias_by_group()

# ==================== 3. 控制台打印验证表 ====================
print("=" * 100)
print("各模型在不同语言下各偏见维度的偏见程度（Bias Degree %）")
print("=" * 100)

for lang in LANGUAGES:
    lang_abbr = LANG_ABBR_MAP[lang]
    print(f"\n--- {lang} ({lang_abbr}) ---")
    header = f"{'Model':>12s}"
    for gn in BIAS_GROUPS:
        header += f" | {gn[:12]:>12s}"
    print(header)
    print("-" * 100)

    for model in MODELS:
        row = f"{model:>12s}"
        for gn in BIAS_GROUPS:
            val = bias_by_group[model][lang_abbr][gn]
            if not np.isnan(val):
                row += f" | {val:>12.3f}%"
            else:
                row += f" | {'N/A':>12s}"
        print(row)

print("\n" + "=" * 100)

# ==================== 4. 构建绘图数据 ====================
data_list = []
for model in MODELS:
    model_data = {'Language': ['CH', 'EN', 'FR', 'JA', 'KO']}
    for group_name in BIAS_GROUPS:
        model_data[group_name] = [
            bias_by_group[model][lang_abbr][group_name]
            for lang_abbr in ['CH', 'EN', 'FR', 'JA', 'KO']
        ]
    data_list.append(model_data)

titles = MODELS

# ==================== 5. 绘图（热力图） ====================
darkblue_to_lightblue = [
    '#FFFFFF',
    '#F5F5F5',
    '#FFF2CC',
    '#F8CECC',
    '#FFD5E7'
]

cmap_name = 'darkblue_to_lightblue_cmap'
darkblue_to_lightblue_cmap = LinearSegmentedColormap.from_list(cmap_name, darkblue_to_lightblue, N=256)
cmap = darkblue_to_lightblue_cmap

# 创建 2x4 布局
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(20, 10), sharex=True, sharey=True)

for i, ax in enumerate(axes.flat):
    df = pd.DataFrame(data_list[i]).set_index('Language').T
    sns.heatmap(df, annot=True, cmap=cmap, cbar=False, ax=ax, linewidths=0.5,
                linecolor='black', fmt=".3f", annot_kws={"size": 7})

    ax.set_title(titles[i], fontsize=12, pad=10)

    if i % 4 != 0:
        ax.set_ylabel('')
        ax.tick_params(left=False)

    ax.tick_params(axis='both', which='major', labelsize=9)

# 移除坐标轴边框
for ax in axes.flat:
    for spine in ax.spines.values():
        spine.set_visible(True)

# 添加 colorbar
fig.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
norm = plt.Normalize(vmin=0, vmax=100)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
fig.colorbar(sm, cax=cbar_ax)
cbar_ax.set_ylabel('Bias degree (%)', fontsize=12)

plt.savefig("natureheatmap.png", dpi=1000)
plt.show()

print(f"\n✅ 热力图已保存至: natureheatmap.png")
