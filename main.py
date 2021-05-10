import gym
import rat_env
import numpy as np
from a3c.actor_critic import Agent
from utils.utils import plot_learning_curve
from gym import wrappers
import pandas as pd

# Bring in the dataset
env_dataset = pd.read_csv("./data/LTE_data.csv")

if __name__ == '__main__':
    #env = gym.make('LunarLander-v2')
    env = gym.make('rat_env_lx-v0', df=[env_dataset])
    agent = Agent(alpha=1e-5, n_actions=env.action_space.n)
    n_games = 1800
    # uncomment this line and do a mkdir tmp && mkdir video if you want to
    # record video of the agent playing the game.
    # env = wrappers.Monitor(env, 'tmp/video', video_callable=lambda episode_id: True, force=True)
    filename = 'cartpole_1e-5_1024x512_1800games.png'

    figure_file = 'plots/' + filename

    best_score = env.reward_range[0]
    score_history = []
    load_checkpoint = False

    if load_checkpoint:
        agent.load_models()

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            if not load_checkpoint:
                agent.learn(observation, reward, observation_, done)
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()

        print('episode ', i, 'score %.1f' %
              score, 'avg_score %.1f' % avg_score)

    if not load_checkpoint:
        x = [i+1 for i in range(n_games)]
        plot_learning_curve(x, score_history, figure_file)
