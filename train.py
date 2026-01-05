from src.env import HareEnv, EnvConfig
from src.obs import encode_discrete
from src.agents.q_table import QTableAgent
from src.utils.io import save_trackers

def run():
    cfg = EnvConfig(start_energy=1000, n_carrots=5, K=200)
    env = HareEnv(cfg, seed=0)
    agent = QTableAgent(n_actions=env.n_actions)

    trackers = []

    for ep in range(2000):

        hare_pos_tracker = []
        wolfs_tracker = []
        carrot_tracker = []

        total_r = 0.0
        carrots = 0
        catches = 0
        steps = 0

        env.reset()
        s = encode_discrete(env.state, cfg.W, cfg.H)

        total_r = 0.0
        while True:
            a = agent.act(s)
            obs, r, done, info = env.step(a)
            hare_pos_tracker.append(obs["hare_pos"])
            wolfs_tracker.append(obs["wolves"])
            carrot_tracker.append(obs["carrots"])
            sp = encode_discrete(env.state, cfg.W, cfg.H)
            agent.update(s, a, r, sp, done)
            s = sp
            total_r += r

            carrots += info.get("carrots_eaten", 0)
            catches += int(info.get("caught", False))
            steps += 1

            if done:
                agent.end_episode()
                trackers.append((hare_pos_tracker, wolfs_tracker, carrot_tracker))
                break

        if ep % 50 == 0:
            if ep % 50 == 0:
                print(
                    ep,
                    "return", total_r,
                    "steps", steps,
                    "carrots", carrots,
                    "catches", catches,
                    "eps", agent.eps
                )
    return trackers

if __name__ == "__main__":
    trackers = run()
    save_trackers(trackers, "outputs/trackers_smth.pkl")