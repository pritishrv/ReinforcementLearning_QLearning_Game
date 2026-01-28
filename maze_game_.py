import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# -------------------------------
# Maze (10×10)
# -------------------------------

maze = np.array([
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [1, 1, 0, 1, 0, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 1, 1, 1, 0, 1, 0, 1, 1, 1],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 1, 1, 1, 1, 0],
    [0, 1, 0, 1, 0, 0, 0, 0, 1, 0],
    [0, 1, 0, 1, 1, 1, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
])

ROWS, COLS = maze.shape
START = (0, 0)
EXIT = (9, 9)

# -------------------------------
# Q-learning Setup (Random Init)
# -------------------------------

Q = np.random.uniform(-1, 1, (ROWS, COLS, 4))

alpha = 0.1
gamma = 0.9
epsilon = 0.4

# Actions: Up Down Left Right
moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def valid(r, c):
    return 0 <= r < ROWS and 0 <= c < COLS and maze[r][c] == 0


def get_reward(pos):
    return 100 if pos == EXIT else -1


# -------------------------------
# Generator: Step-by-step Learning
# -------------------------------

def training_steps():
    global epsilon

    for episode in range(1, 101):

        r, c = START

        for step in range(200):

            # Choose action
            if random.uniform(0, 1) < epsilon:
                action = random.randint(0, 3)
            else:
                action = np.argmax(Q[r, c])

            dr, dc = moves[action]
            nr, nc = r + dr, c + dc

            # Wall hit
            if not valid(nr, nc):
                reward = -10
                nr, nc = r, c
            else:
                reward = get_reward((nr, nc))

            # Q Update
            best_future = np.max(Q[nr, nc])
            Q[r, c, action] += alpha * (
                reward + gamma * best_future - Q[r, c, action]
            )

            r, c = nr, nc

            yield episode, step, (r, c)

            # Exit reached
            if (r, c) == EXIT:
                break

        epsilon *= 0.98


# -------------------------------
# LIVE Animation
# -------------------------------

fig, ax = plt.subplots(figsize=(8, 8))
trainer = training_steps()


def draw(frame):
    episode, step, agent_pos = frame

    ax.clear()

    # Draw maze
    ax.imshow(maze, cmap="gray_r")

    # Exit
    ax.scatter(EXIT[1], EXIT[0], color="green", s=250)

    # Agent Dot (RED)
    ar, ac = agent_pos
    ax.scatter(ac, ar, color="red", s=200)

    # Q-values (small text)
    for i in range(ROWS):
        for j in range(COLS):
            if maze[i][j] == 0:

                up, down, left, right = Q[i, j]

                ax.text(j, i - 0.25, f"{up:.1f}", ha="center", fontsize=5)
                ax.text(j, i + 0.25, f"{down:.1f}", ha="center", fontsize=5)
                ax.text(j - 0.25, i, f"{left:.1f}", va="center", fontsize=5)
                ax.text(j + 0.25, i, f"{right:.1f}", va="center", fontsize=5)

    ax.set_title(f"Episode {episode} | Step {step} | Learning Live")
    ax.set_xticks([])
    ax.set_yticks([])


ani = animation.FuncAnimation(
    fig,
    draw,
    frames=trainer,
    interval=150,
    repeat=False
)

plt.show()
