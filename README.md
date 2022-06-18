# Abalone AI

A homemade reinforcement learning project for game [Abalone](https://en.wikipedia.org/wiki/Abalone_(board_game)).
The goal of this project is to make a **moderately strong** and **moderately fast** game AI within **reasonable time** using a **home PC**.

This project ... 
- **DO NOT** use fancy reinforcement learning algorithms like Q-learning, actor-critic, neither Proximal Policy Optimization.
- Make AI stronger by training self-played games. See below for details.
- Introduce and adopt optimization techniques for MCTS based self-play reinforcement learning.

# Training Abalone AI
This document explains how to train and play against Abalone AI insufficiently.
Please refer this document and read code enough to run code.

Because this code written for Windows PyCharm, they are not passed argument through command line.
Change code directly if you want to use different configuration.

## Train Abalone AI from scratch
You can start training Abalone AI from scratch simply run `run/train_rl_mcts.py`.
Because the code automatically saves the progress in a file, you can suspend and restart training anytime.
However, `run/train_rl_mcts.py` is not tested enough yet.
Please pay attention to whether it is working well.

## Play against Abalone AI
I made a [React project](https://github.com/newmbewb/abalone-ai-react) which provides a user interface for playing Abalone.
The React page communicates with an Abalone AI server over websocket.
The Abalone AI server should be at the same host, or you have to fix the React code to connect to a right Abalone AI server.
You can run Abalone AI server by starting `run/server.py`.
If you want to run your own trained network or agent, change `run/server.py` code.

# Algorithm & Implementation

## How does Abalone AI play game?
Abalone AI agent chooses moves using trained networks and monte-carlo tree search algorithm(MCTS).
The agent has two deep learning models: _policy model_ and _value model_.
_policy model_ finds the best moves for a given game state.
_value model_ evaluates a given game state and tell who the state is in favor of and how much it is.

The move selecting algorithm is very similar to other Monte-Carlo Tree Search, so this page does not explain details,
but Abalone AI agent is different from others in the following aspects.
- Abalone AI agent does not rollout games, but it evaluates the situation using the value model.
- Abalone AI agent does not explore all possible moves, but only _k_ reasonable moves selected by the policy model.
That narrows down explore space and makes the agent faster.
_k_ can be fixed during game, or dynamically selected during move searching time.
- Abalone AI agent calculates(i.e., find the next reasonable moves using policy model, and evaluate the game state using policy model) multiple nodes at once in batch manner.
That makes the agent faster.

## How does Abalone AI trains itself?

Let's assume that we have policy model and value model of the previous generation.
Abalone AI makes better policy model and value model following below steps.

1. An old agent, which works with policy model and value model of previous generation, generates self-play games.
Because the agent uses MCTS, it makes better decision than the old policy model.
Now we have 'game state-move' pairs. They will be the train & test set for the policy model of the next generation.
2. After generate enough 'game state-move' pairs, we train the next policy model using them.
3. Using the newly generated policy model, rollout randomly generated game states.
The rollout repeats 50 times for each game.
The win rate is the score of the given game state.
The 'game state-win rate' will be the train & test set for the value model of the next generation.
4. Train the next value model using the generated train set in the previous step.

Now we have new policy model and value model.

## Optimizations
Because this project targets to train AI on a home PC, we need to optimize codes enough to finish AI training within reasonable time.

### Batching Monte-Carlo Tree Search
When building Monte-Carlo Tree, Abalone AI chooses the next node to travel using UCT algorithm.
Instead of evaluating/rollout chosen node immediately, Abalone AI agent puts it in a queue and find the next node to travel.
When the queue is full, or Abalone AI cannot find more node to evaluate, Abalone AI evaluates them at once.
This allows Abalone AI evaluates nodes in batch manner and makes value model inference faster.
Policy model inference(selecing _k_ children) works in same manner.

### Reusing Monte-Carlo Tree Branches
When selecting a next move, Abalone AI agent builds a Monte-Carlo tree, and select one of children of the root node.
Then the selected child node becomes the root node of the next turn's Monte-Carlo tree.
Therefore, we can reuse the child node instead of building Monte-Carlo tree from scratch.
This optimization is very useful when generating self-play games.

### Avoiding Using numpy.random.choice
`numpy.random.choice` is a good module for randomly choose a next move using the output of a policy model.
Random choose is essential for making train set, but numpy.random.choice takes too much CPU time.
Abalone AI implements lighter methods to choose random moves.

### Augmentation
Abalone game board is rotationally and reflectionally symmetric, so we can make 12 samples from one state-move pair or state-value pair.
This shortens train set generation time.

### Good Starting Point
Using strong AI as the first generation is a good way to boost training.
Although Abalone AI becomes stronger over generation, making a next generation takes too long time.
(It takes 3 days using my Intel i3-7100 CPU)
In this project, I made the first generation AI using MCTS without policy model and value model.
Because only MCTS AI cannot make good game opening, the MCTS AI replays human players record during the first few moves.
It make good staring point to train the first generation using the games generating from MCTS + Human Opening Moves.

### Others
**Coordinate Expression.** Rather than using (x, y) system, this project uses integer index to express the location of points.
This expression method makes calculating relative position between points easier and also makes a program faster very slightly.
Please refer `dlabalone/ablboard.py` and `dlabalone/abltypes.py` for details.

**Batching Game Running.** When evaluating game states, `evaluate_win_probability.py` runs multiple games concurrently and decide next moves of them at once.
This allows large batch inference.

**Small Model.** Training large model for policy and value needs more train set.
Although large model makes AI more complex and stronger, generating train set for it takes too much long time.
Because the goal of this project is to make 'moderately' strong AI, I used small model to make next generation faster.
If you have a system to generate large train set in short time, using larger model is a good idea to make stronger AI.

# Result
I ran this project on my PC with _Intel i3-7100 CPU_ and _RTX3080Ti_.
For each generation I generated enough train set, and it took about 3 days for every generation.

## Generation vs. Generation
#### Generation 1 (width 3, 3000 rounds) vs. MCTS(20000 rounds, temperature 0.01) (Generation 0)
Generation 1: Wins 23, Losses 0, Draws 0
#### Generation 2 (width 3, 3000 rounds) vs. Generation 1 (width 3, 3000 rounds)
Generation 2: Wins 10, Losses 0, Draws 0

Generation 2 is strong enough to play against human player.