import pandas as pd

csv_path = "paper_assets/tables/cf_metrics_summary.csv"
df = pd.read_csv(csv_path)

# 你可以按论文需要重命名列
rename_map = {
    "delta_score_mean": r"$\Delta \hat{y}$",
    "l1_mean": r"$L_{1}$",
    "smooth_mean": r"$L_{\mathrm{smooth}}$",
    "bone_mean": r"$L_{\mathrm{bone}}$",
}
for k,v in rename_map.items():
    if k in df.columns:
        df = df.rename(columns={k:v})

latex = df.to_latex(index=False, float_format="%.4f", escape=False)
print(latex)