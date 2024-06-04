
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numba_progress import ProgressBar
import sys
from numba import njit, prange
from numba.typed import Dict
from numba.core import types, config



@njit
def update(grid, x, y, m):
    if grid[x[0],x[1]]>grid[y[0],y[1]]:

        grid[x[0],x[1]]-= int (m/2*(grid[x[0],x[1]]-grid[y[0],y[1]]))
        grid[y[0],y[1]]+= int (m/2*(grid[x[0],x[1]]-grid[y[0],y[1]]))
    else: 
        grid[y[0],y[1]]-= int (m/2*(grid[y[0],y[1]]-grid[x[0],x[1]]))
        grid[x[0],x[1]]+= int (m/2*(grid[y[0],y[1]]-grid[x[0],x[1]]))


@njit
def agree(grid, x, y, epsilon):
        if grid[x[0],x[1]]>grid[y[0],y[1]]:
            return grid[x[0],x[1]]-grid[y[0],y[1]] <epsilon
        else:
            return grid[y[0],y[1]]-grid[x[0],x[1]]<epsilon
        
@njit(nogil=True)
def relative_agreement_model(
    grid,epsilon,m, steps, max_animation_steps, progress_proxy, animation_interval=1
):
    grids = np.zeros((max_animation_steps,) + tuple(grid.shape), dtype=np.uint8)
    grids[0] = grid
    L = grid.shape[0]

    dt = 0
    animation_step = 1
    t = 0

    # sample single random cell for each step
    #x = np.random.randint(0, L, size=(steps, 2))

    # sample a cardinal direction neighbor for each cell
    #nbr = np.random.randint(0, 4, size=steps)


    ##sample two random cells i and j

    x= np.random.randint(0,L,size=(steps,2))
    y= np.random.randint(0,L,size=(steps,2))


    for i in range(steps):
        progress_proxy.update(1)

        if np.all(grid == grid[0,0]):
            # everyone is in agreement, simulation is done
            break

        # check if opinions agree
        #if agree(grid, x[i], nbr[i]):
            # set opinions of common neighbors
        #    update(grid, x[i], nbr[i])

        #check and possibly modify the opinions of i and j


        if agree(grid, x[i],y[i],epsilon):
            #update opinions of x[i] and y[i]
            update(grid,x[i],y[i],m)


        # animation stuff
        dt = 1
        if (animation_step < max_animation_steps) and (
            (t % animation_interval) == 0):
            grids[animation_step] = grid
            animation_step += 1

        t += dt

    grids[animation_step - 1] = grid
    return grids[:animation_step], t


# Example usage
grid_size,epsilon,m, steps, ani_steps, ani_interval = sys.argv[1:]
grid_size = int(grid_size)
epsilon = int(epsilon)
m =float(m)
steps = int(steps)
ani_steps = int(ani_steps)
ani_interval = float(ani_interval)
initial_grid = np.random.choice(
    np.arange(256),
    size=(grid_size, grid_size) ,
    #p=[p_first] + [(1-p_first)/(n_opinions-1)]*(n_opinions-1),
).astype(np.uint8)

with ProgressBar(total=steps) as progress:
    animation_grid, t = relative_agreement_model(
        initial_grid,epsilon,m, steps, ani_steps, progress, animation_interval=ani_interval
    )


print(f"finished in {t} steps!")
print(f"Generated {len(animation_grid)} frames.")
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
#writer = animation.FFMpegWriter(
#        fps=fps,
        # bitrate=320,
        # extra_args=['-crf', '0', '-preset', 'veryslow', '-c:a', 'libmp3lame']
#)

writer = animation.PillowWriter(
        fps=fps,
)
ani.save(
    f"videos/opinion_animation_L{grid_size}_epsilon{epsilon}_m{m}_steps{steps}_frames{ani_steps}_framedt{ani_interval}.gif",
    dpi=max(128, grid_size / 4),  # ensure that a pixel size is atleast two cells
    writer=writer,
)

#ani.save(f"videos/animation.gif",writer=writer)

plt.show()
