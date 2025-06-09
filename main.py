import gym
from Brain import SACAgent
from Common import Play, Logger, get_params
import numpy as np
from tqdm import tqdm
#import mujoco_py LEGACY
import mujoco


def concat_state_latent(s, z_, n):
    z_one_hot = np.zeros(n)
    z_one_hot[z_] = 1
    return np.concatenate([s, z_one_hot])


if __name__ == "__main__":
    #print("Starting main")
    params = get_params()
    # Create environment and seed it properly
    test_env = gym.make(params["env_name"], render_mode="human") 
    #print("test_env", test_env)
    
    n_states = test_env.observation_space.shape[0]
    #print("n_states", n_states)
    #print("testenv.observation_space is: ", test_env.observation_space)
    n_actions = test_env.action_space.shape[0]
    #print("n_actions", n_actions)
    action_bounds = [test_env.action_space.low[0], test_env.action_space.high[0]]
    #print("action_bounds", action_bounds)
    params.update({"n_states": n_states,
                   "n_actions": n_actions,
                   "action_bounds": action_bounds})
    #print("params:", params)
    test_env.close()
    del test_env, n_states, n_actions, action_bounds

    # Create and seed the environment
    env = gym.make(params["env_name"], render_mode="rgb_array") ####while testing the mode needs to be "rgb_array". while trsining it has to be "human"
    #print("env ::", env)
    #env.seed(params["seed"])
    env.observation_space.seed(params["seed"])
    #print("env.observation_space.seed(params[\"seed\"])", env.observation_space.seed(params["seed"]))
    env.action_space.seed(params["seed"])
    #print("env.action_space.seed(params[\"seed\"])", env.action_space.seed(params["seed"]))

    # Initialize the agent and logger
    p_z = np.full(params["n_skills"], 1 / params["n_skills"])
    #print("p_z is :",p_z)
    agent = SACAgent(p_z=p_z, **params)
    #print("agent ", agent)
    logger = Logger(agent, **params)

    if params["do_train"]:

        if not params["train_from_scratch"]:
            episode, last_logq_zs, np_rng_state, *env_rng_states, torch_rng_state, random_rng_state = logger.load_weights()
            #print("Loaded weights from episode:", episode)
            #print("last_logq_zs:", last_logq_zs)
            #print("np_rng_state:", np_rng_state)
            #print("env_rng_states:", env_rng_states)
            #print("torch_rng_state:", torch_rng_state)
            #print("random_rng_state:", random_rng_state)
            agent.hard_update_target_network()
            min_episode = episode
            # Set the random states for numpy, environment, observation space, action space, and agent
            np.random.set_state(np_rng_state)
            # env.np_random.set_state(env_rng_states[0]) LEGACY
            # env.observation_space.np_random.set_state(env_rng_states[1]) LEGACY
            # env.action_space.np_random.set_state(env_rng_states[2]) LEGACY
            env.np_random.bit_generator.state = env_rng_states[0]
            env.observation_space.np_random.bit_generator.state = env_rng_states[1]
            env.action_space.np_random.bit_generator.state = env_rng_states[2]
            agent.set_rng_states(torch_rng_state, random_rng_state)
            #print("Keep training from previous run.")

        else:
            min_episode = 0
            last_logq_zs = 0
            np.random.seed(params["seed"])
            # Already done earlier, so we don't need to repeat it.
            env.reset(seed=params["seed"])
            # env.seed(params["seed"])
            # env.observation_space.seed(params["seed"])
            # env.action_space.seed(params["seed"])
            #print("Training from scratch.")

        logger.on()
        for episode in tqdm(range(1 + min_episode, params["max_n_episodes"] + 1)):
            # env.render() ##remove this if training is slow!!
            '''
            Say it picks z = 2. This is like saying:

                “Today, I’ll pretend to be personality 2.”

            This skill will be fixed for the entire episode.

            Reason?
            So the agent gets a chance to behave consistently and the system can later tell: how different was Skill 2 from the others?
            '''
            print(f"Episode {episode} / {params['max_n_episodes']}")
            z = np.random.choice(params["n_skills"], p=p_z)
            #print("Chosen skill z:", z)
            # Reset the environment and agent
            obs, _ = env.reset()  # agent spawns into the environment
            #print("Initial state:", obs)
            # Augment the state with the skill z
            state = concat_state_latent(obs, z, params["n_skills"])  # receiving a concatenated list sz
            #print("Augmented state:", state)
            # Initialize the episode reward and logq_zses
            episode_reward = 0
            logq_zses = []

            max_n_steps = min(params["max_episode_len"], env.spec.max_episode_steps)
            for step in range(1, 1 + max_n_steps):
                env.render()
                print(f"==============Step {step} / {max_n_steps}=============")
                action = agent.choose_action(state)  # The policy outputs actions based on state + skill
                # print(f"Step {step} Action: {action}")

                #print("Chosen action:", action)
                # next_obs, reward, done, _, _ = env.step(action)  # This moves the agent a little
                next_obs, reward, done, truncated, info = env.step(action)
                # print(f"Step {step}: Terminated={done}, Truncated={truncated}, Reward={reward}")
                # print("Next Obs:", next_obs)

                #print("Next observation:", next_obs)
                #print("Reward received:", reward)
                #print("Done status:", done)
                # Augment the next observation with the skill z
                next_state = concat_state_latent(next_obs, z, params["n_skills"])  # Augmenting next state
                #print("Augmented next state:", next_state)
                '''
                Why add skill to the state?
                    So the agent can learn different policies for different skills – the network behavior is conditioned on z.
                '''
                #print(f"Storing state, z, done, action, next_state in agent memory.{' ' * 10}State: {state}, z: {z}, Done: {done}, Action: {action}, Next State: {next_state}")
                agent.store(state, z, done, action, next_state)
                logq_zs = agent.train()
                #print("Log likelihood of skill z given next state:", logq_zs)
                # If logq_zs is None, it means the agent didn't train this step, so we use the last value
                if logq_zs is None:
                    logq_zses.append(last_logq_zs)
                else:
                    logq_zses.append(logq_zs)
                episode_reward += reward
                ##print("Episode reward so far:", episode_reward)
                state = next_state # Update the state to the next state
                if done:
                    break
                # Log the total intrinsic reward and the average loglikelihood of the skill 
                # Total intrinsic reward -- measures how distinguishable the skill was during the episode
                # Avg loglikelihood -- measures how confidently the discriminator could classify the skill based on states visited
                                #--- if high -- policy has learned to visit skill specific states -- discriminator is good -- this is the actual training signal that is being maximized 
            logger.log(episode,
                       episode_reward,
                       z,
                       sum(logq_zses) / len(logq_zses),  
                       step,
                       np.random.get_state(),
                    #    env.np_random.get_state(),
                       env.np_random.bit_generator.state,
                    #    env.observation_space.np_random.get_state(),
                    #    env.action_space.np_random.get_state(),
                        env.observation_space.np_random.bit_generator.state,
                        env.action_space.np_random.bit_generator.state,     
                       *agent.get_rng_states(),
                       )

                        


    else:
        print("Evaluating the agent's skills.")
        logger.load_weights() #load the weights of the trained agent
        player = Play(env, agent, n_skills=params["n_skills"])
        player.evaluate()

