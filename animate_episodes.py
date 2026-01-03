import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def load_trackers(path="outputs/trackers.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)

def animate_episode(trackers, episode_idx=0, W=15, H=15, interval=50, save_path=None):
    hare_pos, wolves_frames, carrots_frames = trackers[episode_idx]
    T = len(hare_pos)

    fig, ax = plt.subplots()
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(-0.5, H - 0.5)
    ax.set_aspect("equal")

    # Grid lines (optional)
    ax.set_xticks(np.arange(-0.5, W, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, H, 1), minor=True)
    ax.grid(which="minor", linewidth=0.3)

    # Artists: hare, carrots, calm wolves, hunting wolves
    hare_sc = ax.scatter([], [], s=120, marker="o", label="Hare")
    carrot_sc = ax.scatter([], [], s=60, marker="^", label="Carrots")
    calm_sc = ax.scatter([], [], s=100, marker="s", label="Calm wolves")
    hunt_sc = ax.scatter([], [], s=100, marker="s", label="Hunting wolves")

    title = ax.text(0.02, 1.02, "", transform=ax.transAxes)

    ax.legend(loc="upper right")

    def init():
        hare_sc.set_offsets(np.empty((0, 2)))
        carrot_sc.set_offsets(np.empty((0, 2)))
        calm_sc.set_offsets(np.empty((0, 2)))
        hunt_sc.set_offsets(np.empty((0, 2)))
        title.set_text("")
        return hare_sc, carrot_sc, calm_sc, hunt_sc, title

    def update(t):
        # Hare
        hx, hy = hare_pos[t]
        hare_sc.set_offsets([[hx, hy]])

        # Carrots
        carrots = carrots_frames[t]
        if len(carrots) > 0:
            carrot_sc.set_offsets(np.array(carrots))
        else:
            carrot_sc.set_offsets(np.empty((0, 2)))

        # Wolves (split by hunting flag)
        wolves = wolves_frames[t]  # list of (pos, is_hunting, direction)
        calm = []
        hunt = []
        for (wpos, is_hunting, wdir) in wolves:
            (wx, wy) = wpos
            hunt.append((wx, wy))

        hunt_sc.set_offsets(np.array(hunt) if hunt else np.empty((0, 2)))

        title.set_text(f"Episode {episode_idx} | t = {t+1}/{T}")
        return hare_sc, carrot_sc, calm_sc, hunt_sc, title

    ani = FuncAnimation(fig, update, frames=T, init_func=init, interval=interval, blit=True)

    if save_path:
        # Save as GIF or MP4 depending on extension
        if save_path.lower().endswith(".gif"):
            ani.save(save_path, writer="pillow", fps=max(1, int(1000/interval)))
        else:
            # MP4 requires ffmpeg installed
            ani.save(save_path, writer="ffmpeg", fps=max(1, int(1000/interval)))
        print("Saved:", save_path)

    plt.show()

if __name__ == "__main__":
    trackers = load_trackers("outputs/trackers.pkl")

    # Adjust W/H to your env config if needed
    animate_episode(trackers, episode_idx=1999, W=15, H=15, interval=1000)

    # To save:
    # animate_episode(trackers, episode_idx=0, W=15, H=15, interval=50, save_path="outputs/ep0.gif")
    # animate_episode(trackers, episode_idx=0, W=15, H=15, interval=50, save_path="outputs/ep0.mp4")