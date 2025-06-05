# # from mujoco_py.generated import const LEGACY
# from mujoco_py import GlfwContext LEGACY
import cv2
import numpy as np
import os

# GlfwContext(offscreen=True) LEGACY
'''
This class is responsible for evaluating trained DIAYN skills and recording videos of each skill’s behavior using cv2.VideoWriter

'''

class Play:
    def __init__(self, env, agent, n_skills):
        self.env = env
        self.agent = agent
        self.n_skills = n_skills
        self.agent.set_policy_net_to_cpu_mode()
        self.agent.set_policy_net_to_eval_mode()
        print("Play: Agent policy net set to CPU and eval mode.")
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        print("Play: Video writer fourcc set to XVID.")
        # if not os.path.exists("Vid/"):
        #     os.mkdir("Vid/")
        os.makedirs("Vid", exist_ok=True)

    @staticmethod
    def concat_state_latent(s, z_, n):
        z_one_hot = np.zeros(n)
        z_one_hot[z_] = 1
        s = np.array(s, dtype=np.float32)  # Ensure state is a float32 array
        print(f"Play: Concatenating state {s} with skill {z_} one-hot vector of size {n}.")
        print(f"concatenated state is {np.concatenate([s, z_one_hot])}.")
        return np.concatenate([s, z_one_hot])

    def evaluate(self):
        print("Play: Starting evaluation of skills.")
        for z in range(self.n_skills):
            print(f"Play: Evaluating skill {z}.")
            # Create a video writer for each skill
            video_path = f"Vid/skill{z}.avi"
            video_writer = cv2.VideoWriter(video_path, self.fourcc, 25.0, (250, 250)) # Creates a video file for Skill z at Vid/skill{z}.avi. Frame rate = 50 FPS, frame size = 250x250 pixels.
            s,_ = self.env.reset() ## unpack observation and discard info
            s = self.concat_state_latent(s, z, self.n_skills)
            print(f"Play: Initial state for skill {z}: {s}.")
            print("Shape of state:", s.shape)
            print("Type of state:", type(s))
            episode_reward = 0
            frame_count = 0
            
            for _ in range(self.env.spec.max_episode_steps):
                print(f"===============Play: Skill {z}, Episode step {_}/ {self.env.spec.max_episode_steps}.=====================")
                print(f"Play: Current state: {s}.")
                action = self.agent.choose_action(s) #Feed [state + skill] into the trained agent’s policy. -- only the actor is used here for evaluation
                print(f"Play: Chosen action for skill {z}: {action}.")
                obs, r, terminated, truncated, _ = self.env.step(action) #take the action in the environment
                done= terminated or truncated
                if isinstance(obs, tuple):
                    s_, _ = obs
                else:
                    s_ = obs
                
                s_ = self.concat_state_latent(s_, z, self.n_skills)
                episode_reward += r
                I = self.env.render()  # Render the environment to get the current frame
                if I is None:
                    print(" Rendered frame is None, skipping...")
                elif not isinstance(I, np.ndarray):
                    print("Rendered frame is not an ndarray, skipping...")
                elif len(I.shape) != 3 or I.shape[2] != 3:
                    print(f"Unexpected frame shape: {I.shape}, skipping...")
                else:
                    try:
                        I = cv2.cvtColor(I, cv2.COLOR_RGB2BGR)
                        I = cv2.resize(I, (250, 250))
                        video_writer.write(I)
                        frame_count += 1
                    except Exception as e:
                        print(f"Error writing frame: {e}")

                # I = self.env.render(mode='rgb_array')
                # I = self.env.render()
                # if I is None:
                #     print("!! Rendered frame is None!")
                #     continue  # Skip writing this frame
                # if I is not None:
                #     I = cv2.cvtColor(I, cv2.COLOR_RGB2BGR)
                #     I = cv2.resize(I, (250, 250))
                #     video_writer.write(I) #record the frame of the environment and make a video
                if done:
                    break
                s = s_
            print(f"skill: {z}, episode reward:{episode_reward:.1f}")
            video_writer.release()
            if frame_count == 0:
                print(f"No frames were written for skill {z}. Video may be unreadable: {video_path}")
        self.env.close()
        print("Play: Evaluation completed. Videos saved in Vid/ directory.")
        cv2.destroyAllWindows()
