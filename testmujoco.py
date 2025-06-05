import gym

# Use render_mode="human" for the newer versions of gym
env = gym.make("Hopper-v4", render_mode="human")
env.reset()

for _ in range(10000):
    action = env.action_space.sample()  # Take a random action
    obs, reward, done, truncated, info = env.step(action)  # Take a step in the environment
    
    # Render the environment
    env.render()

    if done or truncated:
        env.reset()
env.close()