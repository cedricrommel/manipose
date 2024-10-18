# %%
import pathlib as Path
import pandas as pd

# %%
mlflow_base_path = Path("/home/crommel/shared/crommel/mlflow_files/mlruns")
protocol1_path_mhmc = mlflow_base_path / "550333576130974034/f078246cece241fda4a78b99a726d306/artifacts/protocol_1_err.csv"
# %%
res = pd.read_csv(protocol1_path_mhmc, index_col=0)
# %%
actions_order = [
    "directions",
    "discussion",
    "eating",
    "greeting",
    "phoning",
    "photo",
    "posing",
    "purchases",
    "sitting",
    "sittingdown",
    "smoking",
    "waiting",
    "walkdog",
    "walking",
    "walktogether",
    "average",
]
# %%
to_report = res.loc[actions_order, "oracle mpjpe"].to_frame().T
# %%
to_report.to_latex(float_format="%.1f")

# %%
to_report_ave = res.loc[actions_order, "mpjpe"].to_frame().T
# %%
to_report_ave.to_latex(float_format="%.1f")

# %%
