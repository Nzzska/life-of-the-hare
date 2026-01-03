from src.env import HareEnv, EnvConfig
from src.obs import encode_discrete
from src.agents.q_table import QTableAgent
import numpy as np

def evaluate(agent, env, episodes=100):
    agent.eps = 0.0
    returns = []
    carrots = []
    steps = []

    for _ in range(episodes):
        env.reset()
        s = encode_discrete(env.state, env.cfg.W, env.cfg.H)
        total = 0
        t = 0
        c = 0
        while True:
            a = agent.act(s)
            _, r, done, info = env.step(a)
            s = encode_discrete(env.state, env.cfg.W, env.cfg.H)
            total += r
            t += 1
            c += info["carrots_eaten"]
            if done:
                break
        returns.append(total); carrots.append(c); steps.append(t)

    print("avg return", np.mean(returns))
    print("avg carrots", np.mean(carrots))
    print("avg steps", np.mean(steps))

if __name__ == "__main__":
    cfg = EnvConfig()
    env = HareEnv(cfg, seed=1)
    agent = QTableAgent(n_actions=env.n_actions)
    # load trained Q here if you implement save/load
    evaluate(agent, env)
