# Author: Piyush Sharma
import sys, json
sys.path.insert(0, "code")
from multilevel_analysis import run_multilevel_analyses
from data_collection import get_benchmark_data, get_benchmark_metadata
from statistical_analysis import normalize_scores

df = get_benchmark_data()
bm = get_benchmark_metadata()
df_norm = normalize_scores(df, bm)
ml = run_multilevel_analyses(df_norm)

out = []
out.append("PRIMARY MODEL (M4)")
pm = ml["primary_model"]
out.append(f"s2b={pm['sigma2_between']:.4f}")
out.append(f"s2w={pm['sigma2_within']:.4f}")
out.append(f"ICC={pm['icc']:.4f}")
out.append(f"AIC={pm['aic']:.2f}")
out.append(f"BIC={pm['bic']:.2f}")
out.append("")

out.append("VARIANCE DECOMPOSITION")
for _, r in ml["variance_decomposition"].iterrows():
    out.append(f"{r['model']}: s2b={r['sigma2_between']:.4f} ICC={r['icc']:.3f} R2b={r['r_sq_between']:.3f}")
out.append("")

out.append("LRT")
for k, v in ml["lrt"].items():
    out.append(f"{k}: chi2={v['chi2']:.2f} p={v['p']:.4f}")
out.append("")

out.append("SE SENSITIVITY")
for _, r in ml["se_sensitivity"].iterrows():
    out.append(f"f={r['se_floor']:.3f} d={r['pooled_d']:.4f} I2={r['I_sq']:.1f} slp={r['mr_slope']:.4f} p={r['mr_slope_p']:.4f}")
out.append("")

out.append("MODERATORS")
for _, r in ml["moderator_table"].iterrows():
    out.append(f"[{r['model']}] {r['parameter']}: {r['coefficient']:.4f} [{r['ci_lower']:.3f},{r['ci_upper']:.3f}] p={r['p']:.4f}")

with open("paper/tables/ml_clean.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(out))
print("Done")
