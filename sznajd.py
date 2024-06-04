
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numba_progress import ProgressBar
import sys
from numba import njit, prange
from numba.typed import Dict
from numba.core import types, config


RNEIGHBORS = np.array(
    ((-1, 0), (0, -1), (1, -1), (2, 0), (1, 1), (0, 1)),
    dtype=np.int64,
)
DIRECTIONS = np.array([[0,-1], [-1,0], [0,1], [1,0]])

@njit
def update(grid, x1, nbr_direction):
    L = grid.shape[0]
    is_after, is_horizontal = np.divmod(nbr_direction, 2)
    flip = (2*is_after - 1)
    nbrs = RNEIGHBORS*flip
    if not is_horizontal:
        nbrs = nbrs[:,::-1]
    idx = (nbrs + x1[None,:]) % L
    # print(idx)
    for i in idx:
        grid[i[0], i[1]] = grid[x1[0], x1[1]]

@njit
def agree(grid, x1, nbr_direction):
        L = grid.shape[0]
        d = DIRECTIONS[nbr_direction]
        dx = d[0]
        dy = d[1]
        return grid[x1[0], x1[1]] == grid[(x1[0]+dx)%L,(x1[1]+dy)%L]

@njit(nogil=True)
def sznajd_model(
    grid, steps, max_animation_steps, progress_proxy, animation_interval=1
):
    grids = np.zeros((max_animation_steps,) + tuple(grid.shape), dtype=np.uint16)
    grids[0] = grid
    L = grid.shape[0]

    dt = 0
    animation_step = 1
    t = 0


    for i in range(steps):
        progress_proxy.update(1)

        # sample single random cell for each step
        x = np.random.randint(0, L, size=2)

        # sample a cardinal direction neighbor for each cell
        nbr = np.random.randint(0, 4)

        if np.all(grid == grid[0,0]):
            # everyone is in agreement, simulation is done
            break

        # check if opinions agree
        if agree(grid, x, nbr):
            # set opinions of common neighbors
            update(grid, x, nbr)

        # animation stuff
        dt = 1
        if (animation_step < max_animation_steps) and (
            (t % animation_interval) == 0):
            grids[animation_step] = grid
            animation_step += 1

        t += dt

    grids[animation_step - 1] = grid
    return grids[:animation_step], t

@njit(nogil=True, parallel=True)
def run_parallel(
    n_simulations, progress, L, p_first, steps, max_animation_steps = 1, animation_interval=-1
):
    time = np.empty(n_simulations, dtype=np.int32)

    for n in prange(n_simulations):
        grid = np.random.uniform(low=0, high=1, size=(L,L))
        grid = np.where(grid < p_first,
                        np.zeros((1,1), dtype=np.uint16),
                        1 + ((grid-p_first)/(1-p_first)*(n_opinions-1)).astype(np.uint16)
                        )
        anim_grid, t = sznajd_model(grid, steps, max_animation_steps, progress, animation_interval=animation_interval)
        time[n] = t
    return np.zeros((1, L, L), dtype=np.uint16), time

# Example usage
grid_size, n_opinions, p_first, steps, ani_steps, ani_interval, n_simulations = sys.argv[1:]
grid_size = int(grid_size)
p_first = float(p_first)
n_opinions = int(n_opinions)
steps = int(steps)
ani_steps = int(ani_steps)
ani_interval = float(ani_interval)
n_simulations = int(n_simulations)
initial_grid = np.random.choice(
    np.arange(n_opinions),
    size=(grid_size, grid_size),
    p=[p_first] + [(1-p_first)/(n_opinions-1)]*(n_opinions-1),
).astype(np.uint16)

if n_simulations == 1:
    with ProgressBar(total=steps) as progress:
        animation_grid, t = sznajd_model(
            initial_grid, steps, ani_steps, progress, animation_interval=ani_interval
        )
    print(f"finished in {t} steps!")
    print(f"Generated {len(animation_grid)} frames.")
else:

    probs = [0.5, 0.505, 0.51, 0.55, 0.6]
    for n, p in enumerate(probs):
        with ProgressBar(total=n_simulations*steps) as progress:
            animation_grid, times = run_parallel(n_simulations, progress, grid_size, p_first, steps)
        mean_time = sum(times)/len(times)
        print(f"average steps per simulation {mean_time}")
        plt.hist(times, bins=32, alpha = 0.5, density=True, label = f'p1 = {p}, mean {mean_time:.0f}')
    t = times[0] 
    plt.legend()
    plt.xlabel("Simulation end step")
    plt.ylabel("Binned probability")
# Create a figure and axis
fig1, ax1 = plt.subplots(figsize=(8, 8))
ax1.imshow(animation_grid[-1], cmap = "viridis")
ax1.set_axis_off()  # Hide the axis
fig1.tight_layout()

# Create animation window
fig, ax = plt.subplots(figsize=(8, 8))
fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
im = ax.imshow(animation_grid[0], animated=True, cmap = "viridis")
ax.set_axis_off()
fps = 20


def update(frame):
    colors = animation_grid[frame]
    im.set_array(colors)
    return [im]


ani = animation.FuncAnimation(
    fig,
    update,
    frames=len(animation_grid),
    interval=1000 / fps,
    blit=False,
    repeat=True,
)
writer = animation.FFMpegWriter(
        fps=fps,
        # bitrate=320,
        # extra_args=['-crf', '0', '-preset', 'veryslow', '-c:a', 'libmp3lame']
)
ani.save(
    f"videos/opinion_animation_L{grid_size}_N{n_opinions}_pfirst{p_first}_steps{steps}_frames{ani_steps}_framedt{ani_interval}.mp4",
    dpi=max(128, grid_size / 4),  # ensure that a pixel size is atleast two cells
    writer=writer,
)


plt.show()
