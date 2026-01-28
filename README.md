Reinforcement Learning Maze Solver

------------------------------------------Project Overview-------------------------------------

This project is a small reinforcement learning puzzle simulation where an agent (shown as a red dot) learns how to escape a maze.

The maze is represented as a 2D grid, and the agent gradually learns the best path to the exit using Q-learning. Over many iterations, it improves its action choices based on rewards and penalties.

--------------------------------------------How It Works---------------------------------------

The maze is a 10×10 matrix

0 = open path

1 = wall

The agent starts in the top-left corner.

At each step, the agent can move:

Up

Down

Left

Right

Each cell stores 4 Q-values (one for each possible direction).

The agent updates these values based on:

Small penalty for every move

Larger penalty for hitting walls

Big reward for reaching the exit

Over time, the Q-values improve and the agent learns the shortest escape route.

------------------------------------Setup Instructions-------------------------------------
1. Create a virtual environment (recommended)
python -m venv rl_env
source rl_env/bin/activate   # Mac/Linux
rl_env\Scripts\activate      # Windows

2. Install required libraries
pip install numpy matplotlib

-----------------------------------How to Run the Project-----------------------------------

Simply run the Python file:

python maze_qlearning.py


This will open a window showing:

The maze layout

The red dot moving and learning

Q-values updating live inside each cell

The exit marked in green

----------------------------------- Adjust Animation Speed------------------------------

To slow down the movement, increase the interval:

interval=500


Higher values = slower animation.

That’s it — the agent will start randomly and eventually learn how to escape the maze.
