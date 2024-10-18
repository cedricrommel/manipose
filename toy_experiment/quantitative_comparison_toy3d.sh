#!/bin/bash

declare -a seeds=(42 43 44 45 46)

for seed in "${seeds[@]}"
do
    echo "Training and evaluating MLP with seed ${seed}..."
    python main.py model.arch=mlp +data=3D_setup +train=3D_setup run.seed=${seed} run.name=3D_mlp_seed_${seed}

    echo "Training and evaluating Constrined MLP with seed ${seed}..."
    python main.py model.arch=constrained +data=3D_setup +train=3D_setup run.seed=${seed} run.name=3D_constrained_seed_${seed}

    echo "Training and evaluating ManiPose with seed ${seed}..."
    python main.py model.arch=constrained_rmcl +data=3D_setup +train=3D_setup run.seed=${seed} run.name=3D_manipose_seed_${seed}
done