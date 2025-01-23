---
layout: post
category: RL
---

This is the first log in the development of cooperative machines. The agent will play a game called *Paint Your Partner* with its human partner. Eventually, the agent will be composed of a layered architecture of Reinforcement Learning units. However, this post focuses on training an agent to play the game alone, using a custom environment created with [Gymnasium](https://gymnasium.farama.org/) and [Stable Baseline 3](https://stable-baselines3.readthedocs.io/en/master/index.html).

## Table of contents

- [Custom Environment](#custom-environment)
- [Reward Mechanism](#reward-mechanism)
- [Training](#training)
- [Validation](#policy)
- [Thoughts](#thoughts)

## [Custom Environment](#custom-environment)

### Paint Your Partner

The custom environment I created to train the RL agent is a game environment called **Paint Your Partner**. The goal of this game is simple: all you need to do is <mark>touch the goal tile</mark> while <mark>matching your body color to the color of the goal tile.</mark>

![pyp-game](https://raw.githubusercontent.com/nik-pitts/machinelearning601/master/_images/2025-01-19-pyp-game.jpg)

## [Reward Mechanism](#reward-mechanism)

Designing the *reward mechanism* was the most difficult task. At first, I tried a highly defined version of the reward system; however, I eventually realized that this isn’t how RL works. If we design the reward system too tightly, there will be no need to use RL as the agent’s behavior algorithm. Therefore, after numerous attempts, I decided to design the reward mechanism in a more open manner.

'''
    def step(self, action):
        info = {
            "is_success": False,
            "TimeLimit.truncated": False,
        }
		
        self.steps_taken += 1
        max_steps = self.grid_size ** 2

        # Handle movement
        if action == 0 and self.agent_pos[0] > 0:  # Move up
            self.agent_pos[0] -= 1
        elif action == 1 and self.agent_pos[0] < self.grid_size - 1:  # Move down
            self.agent_pos[0] += 1
        elif action == 2 and self.agent_pos[1] > 0:  # Move left
            self.agent_pos[1] -= 1
        elif action == 3 and self.agent_pos[1] < self.grid_size - 1:  # Move right
            self.agent_pos[1] += 1

        reward = 0  # Initialize reward

        # Check for water tiles
        if self.water_tiles[self.agent_pos[0], self.agent_pos[1]] == 1:
            self.agent_color = Pallete.TRANSPARENT

        # Check for painting
        if self.agent_color != self.goal_color:
            for color_idx, chip_present in enumerate(self.color_chips[self.agent_pos[0], self.agent_pos[1]]):
                if chip_present:
                    new_color = self._combine_colors(self.agent_color, self.cmy[color_idx])
                    self.agent_color = new_color

        # Check if the agent has reached the goal with the correct color
        terminated = np.array_equal(self.agent_pos, self.goal_pos) and self.agent_color == self.goal_color
        if terminated:
            reward += 100  # High reward for completing the objective
            info['is_success'] = True
        elif np.array_equal(self.agent_pos, self.goal_pos):  # Reaching the goal without the correct color
            pass # Eventually, I deleted this reward as well.
        elif self.agent_color == self.goal_color:  # Matching color without reaching the goal
            reward += 1

        # Truncation logic
        truncated = self.steps_taken >= max_steps
        if truncated:
            info["TimeLimit.truncated"] = True
            print(f"Episode truncated after {self.steps_taken} steps. No success achieved.")

        # Debugging information for termination
        if terminated:
            print(f"Agent Position: {self.agent_pos}, Goal Position: {self.goal_pos}")
            print(f"Agent Color: {self.agent_color}, Goal Color: {self.goal_color}")
            print(f"Terminated: {terminated}")

        return self._get_obs(), reward, terminated, truncated, info
'''

## [Training](#training)

After tuning the reward mechanism to be as simple as possible, constructing the training algorithm was straightforward. **Ensure sufficient training time!** I monitored the training progress using TensorBoard. Although the progress was very slow, the trend of the episode reward mean and success rate was promising.

'''
    def train_agent(self):
        # Wrap the environment
        env = make_vec_env(
            lambda: SB3CompatibleEnv(PaintYourPartnerGymEnv(render_mode=None)),
            n_envs=1
        )

        Create the PPO model with TensorBoard logging
        model = PPO(
            "MultiInputPolicy",
            env,
            verbose=1,
            ent_coef=0.05,
            tensorboard_log=self.tb_logs_dir
        )

        # Training loop
        TIMESTEPS = 10000
        episodes = 200
        for ep in range(1, episodes):
            model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO_1")
            model.save(f"{self.model_logs_dir}PPO_1/{TIMESTEPS*ep}")

        # Save the final model
        model.save(f"{self.model_logs_dir}PPO_1/trained_model_2.zip")
        print(f"Model saved to {self.model_logs_dir}PPO_1/trained_model_2.zip.")

        # Close the environment
        env.close()
'''

![tb-trend-monitoring](https://raw.githubusercontent.com/nik-pitts/machinelearning601/master/_images/2025-01-19-tb-trend-monitoring.png)

### Tensorboard Report

![tb-trend-monitoring](https://raw.githubusercontent.com/nik-pitts/machinelearning601/master/_images/2025-01-19-tb-log-all.png)

## [Validation](#validation)

Here's the validation video of the resulting agent.

### First version of PYP RL agent

![first-version-pyp-rl](2025-01-19-pyp-validation-1.mp4)

### Improved version of PYP RL agent

![improved-version-pyp-rl](2025-01-19-pyp-validation-2.mp4)

## [Thoughts](#thoughts)

It turned out that the combination of the simplest reward mechanism and extended training time aligns well with the concept of RL. At first, I fine-tuned the reward mechanism, but the more I adjusted it, the more unexpected behaviors I encountered. Machines are remarkably good at uncovering the pitfalls of human logic.