# Super Mario RL Agent (DQN)

Train and play a Deep Q-Network agent on *Super Mario Bros.* using `gym-super-mario-bros`, `nes_py`, and PyTorch.

## Tested setup
- **OS:** Windows
- **Python:** 3.11
- **Key pins (to avoid known breaks):**
  - `numpy==1.26.4`
  - `opencv-python-headless<4.12` *(NumPy 2.x can trigger an `OverflowError` in `nes_py`.)*

## Setup

1. **Create and activate a virtual environment**
   ```powershell
   python -m venv mario311
   mario311\Scripts\Activate.ps1
   ```

2. **Upgrade installer tools**
   ```powershell
   python -m pip install --upgrade pip
   ```

3. **Install dependencies**
   ```powershell
   pip install -r requirements.txt
   ```

> In VS Code: **Ctrl+Shift+P → “Python: Select Interpreter” →** choose `.\mario311\Scripts\python.exe`.

## Train

```powershell
python train.py
```
- Checkpoints will be saved to the `checkpoints/` folder (created automatically on first run).

## Play (using your own latest checkpoint)

```powershell
python play.py
```
- `play.py` loads the most recent `.pt` file from `checkpoints/`.
- You can adjust the exploration rate (`EPSILON`) inside `play.py`.

## Use the provided winning snapshot

1. Complete **Setup** above.  
2. Extract the archive included in the repo (e.g., right-click → **Extract Here**, or with 7‑Zip CLI):
   ```powershell
   7z x "winning_model_snapshot(6100 episodes).7z" -o.
   ```
3. Rename the extracted model to:
   ```
   mario_dqn_ep6100.pt
   ```
4. Place it into the `checkpoints/` directory (create it if missing):
   ```powershell
   mkdir checkpoints  # only if it doesn't exist
   move .\mario_dqn_ep6100.pt .\checkpoints\
   ```
5. Run:
   ```powershell
   python play.py
   ```

## Plot training progress

```powershell
python plot_progress.py
```
- If `checkpoints/` doesn’t exist yet, run `train.py` once to create it.

## Repo layout

```
agent.py
evaluate.py
play.py
plot_progress.py
requirements.txt
train.py
wrappers.py
(checkpoints/)            # created at runtime
winning_model_snapshot(6100 episodes).7z
```
