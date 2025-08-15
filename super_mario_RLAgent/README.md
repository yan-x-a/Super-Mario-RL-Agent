# Super Mario RL Agent (DQN)

Train and play a Deep Q-Network agent on Super Mario Bros using `gym-super-mario-bros`, `nes_py`, and `PyTorch`.

## Tested setup
- OS: Windows (PowerShell commands below)
- Python: 3.11
- Key versions pinned to avoid known breaks:
  - `numpy==1.26.4`
  - `opencv-python-headless<4.12`  
  (NumPy 2.x can trigger an `OverflowError` in `nes_py`.)


## How to set up

# 1) Create and activate a clean venv in terminal 
python -m venv mario311
mario311\Scripts\Activate.ps1

# 2) Upgrade installer tools
python -m pip install --upgrade pip

# 3) Install project deps
pip install -r requirements.txt

# 4) Run train.py or play.py (needs a checkpoint) with python interpreter
python train.py

## How to run the winning snapshot

# 1) Complete the setup instructions 

# 2) Extract the winning_model snapshot

# 3) Rename to "mario_dqn_ep6100.pt"

# 4) Place snapshot into checkpoints directory 
(You might need to run train.py once if this folder doesn't exist)

# 5) Run play.py

## How to show progress

# 1) Complete the setup instructions 

# 2) Run plot_progress.py 
(You might need to run train.py once for checkpoints folder to be created)

