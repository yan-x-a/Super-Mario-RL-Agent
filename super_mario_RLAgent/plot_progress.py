import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("progress.csv")

plt.plot(df["episode"], df["reward"], label="episode reward", alpha=0.4)
df["reward"].rolling(100).mean().plot(label="rolling avg (100)", linewidth=2)

plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Mario DQN learning curve")
plt.legend()
plt.tight_layout()
plt.show()
