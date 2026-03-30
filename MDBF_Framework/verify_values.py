"""Verify computed values for inclusion in paper tables."""
from compute_results import *

results = run_pipeline()

print("\n\n" + "=" * 60)
print("VALUES FOR PAPER TABLES")
print("=" * 60)

print("\n--- Table 4: Dimension Scores ---")
dim_keys = list(results["dimension_scores"].keys())
for i, model in enumerate(MODELS):
    scores = [results["dimension_scores"][k][i] for k in dim_keys]
    overall = results["overall_scores"][i]
    print(f"{model:20s} D1={scores[0]:5.1f}  D2={scores[1]:5.1f}  D3={scores[2]:5.1f}  D4={scores[3]:5.1f}  D5={scores[4]:5.1f}  Overall={overall:5.1f}")

print("\n--- Table 5: PGR Values ---")
for dim, vals in results["pgr"].items():
    print(f"{dim:35s} Prop={vals['prop_mean']:5.1f}  Open={vals['open_mean']:5.1f}  PGR={vals['pgr']:5.1f}%")

print("\n--- Table 6: Statistical Tests ---")
for t in results["statistical_tests"]:
    sig = "*" if t["significant"] else ""
    print(f"{t['model_a']:20s} vs {t['model_b']:20s}  W={t['w_stat']:5.1f}  p={t['p_value']:.3f} {sig}")

print("\n--- Correlation Matrix ---")
dim_names, corr = results["correlation"]
short = ["D1", "D2", "D3", "D4", "D5"]
for i in range(len(short)):
    row = f"{short[i]:4s}"
    for j in range(len(short)):
        row += f"  {corr[i][j]:6.3f}"
    print(row)
