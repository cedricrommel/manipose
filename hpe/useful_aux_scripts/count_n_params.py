# %%
import pickle
from pathlib import Path

from mh_so3_hpe.architectures import RMCLManifoldMixSTE, ManifoldMixSTE, MixSTE

# %% FETCH DATASET SKELETON

data_dir = Path("/home/crommel/shared/crommel/h36m_data")
preproc_dataset_path = data_dir / (
    "preproc_data_3d_h36m_17_mh_so3_hpe.pkl"
)

if preproc_dataset_path.exists():
    print("==> Loading preprocessed dataset...")
    with open(preproc_dataset_path, "rb") as f:
        dataset = pickle.load(f)

# %%
ours = RMCLManifoldMixSTE(
    skeleton=dataset.skeleton
)
constrained = ManifoldMixSTE(
    skeleton=dataset.skeleton
)
mixste = MixSTE()
# %%
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
# %%
print("MixSTE:", count_parameters(mixste))
print("Constrained:", count_parameters(constrained))
print("ManiPose:", count_parameters(ours))
# %%
