#!/bin/bash

declare -a seeds=(42 43 44 45 46)

for seed in "${seeds[@]}"
do
    echo "Training and evaluating MLP with seed ${seed}..."
    python main.py model.arch=mlp +train=mlp_hard2 data.scenario=hard-2 run.seed=${seed} run.name=mlp_seed_${seed}

    echo "Training and evaluating Constrined MLP with seed ${seed}..."
    python main.py model.arch=constrained +train=constrained_hard2 data.scenario=hard-2 run.seed=${seed} run.name=constrained_seed_${seed}

    echo "Training and evaluating ManiPose with seed ${seed}..."
    python main.py model.arch=constrained_rmcl +train=rmcl_constrained_hard2 data.scenario=hard-2 run.seed=${seed} run.name=manipose_seed_${seed}
done