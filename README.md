## Overview
This project presents a deep reinforcement learning approach to incentive-based demand response in smart grids.
The agent file is named `dqn`, but it actually implements the **Double Deep Q-Network (DDQN)** algorithm.

## Requirements
- Python 3.10+

## Run the project

```bash
python -m venv .venv
# Windows
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt

python main.py

python test.py --model_path "results/ddqn_<timestamp>/dqn_final_model.pth"
