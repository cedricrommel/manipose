#!/bin/bash

echo "Training and evaluating MLP..."
python main.py model.arch=mlp +train=mlp_hard2 

echo "Training and evaluating Constrined MLP..."
python main.py model.arch=constrained +train=constrained_hard2

echo "Training and evaluating ManiPose..."
python main.py model.arch=constrained_rmcl +train=rmcl_constrained_hard2
