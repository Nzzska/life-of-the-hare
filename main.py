from src.env import HareEnv, EnvConfig

if __name__ == "__main__":
    env = HareEnv(EnvConfig(), seed=0)
    obs = env.reset()
    total = 0

    for _ in range(200):
        a = env.rng.integers(0, env.n_actions)
        obs, r, done, info = env.step(int(a))
        total += r
        if done:
            obs = env.reset()

    print("ok, total reward:", total)
