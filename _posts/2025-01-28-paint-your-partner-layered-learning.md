---
layout: post
category: RL
---

In my last post, I trained a Paint Your Partner agent using reinforcement learning. Specifically, I provided the agent with all game information as a single, large chunk of observation data. With sufficient training time, the agent successfully learned to play the game.

However, in today’s post, I’ll explore an alternative approach—layered learning. In this paradigm, the agent learns to handle individual subtasks separately before being exposed to the complete environment. For the PYP environment, the key subtasks are:

	1.	Wandering
	2.	Painting
	3.	Moving to the goal

In this post, I’ll discuss how the agent is trained to perform each of these tasks. However, a crucial aspect of layered learning remains: integrating these layers to form a cohesive and complex behavior.

## Table of contents

- [Wandering](#wandering)
- [Painting](#painting)
- [Moving to the goal](#moving-to-the-goal)

## [Wandering](#wandering)

At first, I thought training a wandering behavior would be easy. However, I soon realized that training a purposeless behavior, like wandering, was even more conceptually challenging than training a complex behavior such as playing Paint Your Partner from scratch.

The difficulty stemmed from two key issues. First, because wandering lacks a clear **goal**, I had to carefully design an appropriate reward system. Second, I needed to balance **exploitation and exploration** effectively. Wandering is fundamentally an exploratory behavior, but since the agent is playing the game, its wandering actions still need to be effective in some way.

Below is the code that successfully emulates the wandering behavior. Basically, I rewarded agent if they successfully visited all grids in a given amounts of steps - impotantly more than the size of the grid, rewarding them if they've visited different positions from where they already visited.

```
    def step(self, action):

        new_position = self.agent_pos.copy()
        if action == 0: new_position[1] += 1  # Up
        elif action == 1: new_position[1] -= 1  # Down
        elif action == 2: new_position[0] -= 1  # Left
        elif action == 3: new_position[0] += 1  # Right

        reward = 0

        if 0 <= new_position[0] < self.grid_size and 0 <= new_position[1] < self.grid_size:
            self.agent_pos = new_position
            if tuple(self.agent_pos) not in self.visited_positions:
                reward = 1  # Reward for new tile
                self.visited_positions.add(tuple(self.agent_pos))

        # Check termination condition (e.g., excessive looping)
        terminated = len(self.visited_positions) == self.grid_size**2
        truncated = self.steps_taken >= self.grid_size**3

        if terminated:
            self.info["is_success"] = True
            reward += 100
        elif truncated:
            self.info["TimeLimit.truncated"] = True

        self.steps_taken += 1

        return self._get_obs(), reward, terminated, truncated, self.info
```

### Wandering Behavior

<video width="360" height="360" controls>
  <source src="https://raw.githubusercontent.com/nik-pitts/machinelearning601/master/_images/2025-01-28-ll-wandering.mp4" type="video/mp4">
  Improved.
</video>

### Tensorboard Report

![tb-trend-monitoring](https://raw.githubusercontent.com/nik-pitts/machinelearning601/master/_images/2025-01-28-wandering-tb.png)

## [Painting](#painting)

Interestingly, painting is more closely related to <mark>planning</mark> than simply taking actions. For the agent to match its body color to the goal color, the first priority is to devise an effective strategy—a **sequence** of coloring steps. Only after that does the action of painting take place, guiding the agent toward the color chip, which serves as an interim goal.

Therefore, this layer functions as the agent’s mental planning ability for painting.

```
self.action_space = spaces.Discrete(4)  # 0: cyan, 1: magenta, 2: yellow, 3: transparent, 4: do nothing
```

```
    def step(self, action):
        info = {
            "is_success": False,
            "TimeLimit.truncated": False,
        }
        self.steps_taken += 1  # Increment step counter
        max_steps = self.grid_size ** 2  # Example: step limit based on grid size

        # Handle painting action
        if action == 0:  # Combine with Cyan
            self.agent_color = self._combine_colors(self.agent_color, self.cmytx[action])
        elif action == 1:  # Combine with Magenta
            self.agent_color = self._combine_colors(self.agent_color, self.cmytx[action])
        elif action == 2:  # Combine with Yellow
            self.agent_color = self._combine_colors(self.agent_color, self.cmytx[action])
        elif action == 3:  # Combine with Transparent
            self.agent_color = Pallete.TRANSPARENT
        elif action == 4:
            self.agent_color = self.agent_color

        reward = 0  # Initialize reward

        # Check if the agent has reached the goal with the correct color
        terminated = self.agent_color == self.goal_color
        if terminated:
            reward += 10  # High reward for completing the objective
            #print(f"Agent Color: {self.agent_color}, Goal Color: {self.goal_color}")
            #print(f"Terminated: {terminated}")
            info['is_success'] = True

        # Truncation logic
        truncated = self.steps_taken >= max_steps
        if truncated:
            info["TimeLimit.truncated"] = True
            #print(f"Agent Color: {self.agent_color}, Goal Color: {self.goal_color}")
            #print(f"Episode truncated after {self.steps_taken} steps. No success achieved.")

        reward -= 0.1

        return self._get_obs(), reward, terminated, truncated, info
```

### [Painting Logs]

Below is evaluation log of painting layer. As you can see, the agent learned how to plan the color most effectively throughout.

```
Episode 4: Reward = 9.9, Steps = 1
[<Pallete.CYAN: 0>]
Goal Color: Pallete.CYAN_MAGENTA_YELLOW, Initial Agent Color: Pallete.CYAN_YELLOW
Episode 5: Reward = 9.9, Steps = 1
[<Pallete.MAGENTA: 1>]
Goal Color: Pallete.CYAN_YELLOW, Initial Agent Color: Pallete.CYAN
Episode 6: Reward = 9.9, Steps = 1
[<Pallete.YELLOW: 2>]
Goal Color: Pallete.CYAN_MAGENTA, Initial Agent Color: Pallete.MAGENTA_YELLOW
Episode 7: Reward = 9.700000000000001, Steps = 3
[<Pallete.TRANSPARENT: 7>, <Pallete.CYAN: 0>, <Pallete.MAGENTA: 1>]
Goal Color: Pallete.CYAN_YELLOW, Initial Agent Color: Pallete.YELLOW
Episode 8: Reward = 9.9, Steps = 1
[<Pallete.CYAN: 0>]
Goal Color: Pallete.CYAN_MAGENTA, Initial Agent Color: Pallete.MAGENTA_YELLOW
Episode 9: Reward = 9.700000000000001, Steps = 3
[<Pallete.TRANSPARENT: 7>, <Pallete.CYAN: 0>, <Pallete.MAGENTA: 1>]
Goal Color: Pallete.CYAN_YELLOW, Initial Agent Color: Pallete.MAGENTA
Episode 10: Reward = 9.700000000000001, Steps = 3
[<Pallete.TRANSPARENT: 7>, <Pallete.YELLOW: 2>, <Pallete.CYAN: 0>]
Evaluation over 10 episodes:
Mean Reward: 9.81
Success Rate: 100.00%
Truncation Rate: 0.00%
```


## [Moving to the goal](##moving-to-the-goal)

This layer was the easiest sub layer to train. The observation the agent needs is its current position and the row, column of the goal.

```
    def step(self, action):

        if action == 0 and self.agent_pos[0] > 0:  # Move up
            self.agent_pos[0] -= 1
        elif action == 1 and self.agent_pos[0] < self.grid_size - 1:  # Move down
            self.agent_pos[0] += 1
        elif action == 2 and self.agent_pos[1] > 0:  # Move left
            self.agent_pos[1] -= 1
        elif action == 3 and self.agent_pos[1] < self.grid_size - 1:  # Move right
            self.agent_pos[1] += 1

        reward = 0

        # Check termination condition (e.g., excessive looping)
        terminated = np.array_equal(self.agent_pos, self.goal_pos)
        truncated = self.steps_taken >= self.grid_size**3

        if terminated:
            self.info["is_success"] = True
            reward += 100
        elif truncated:
            self.info["TimeLimit.truncated"] = True

        self.steps_taken += 1

        return self._get_obs(), reward, terminated, truncated, self.info
```

### Moving to the goal behavior

<video width="360" height="360" controls>
  <source src="https://raw.githubusercontent.com/nik-pitts/machinelearning601/master/_images/2025-01-28-ll-goaling.mp4" type="video/mp4">
  Improved.
</video>

### Tensorboard Report

![tb-trend-monitoring](https://raw.githubusercontent.com/nik-pitts/machinelearning601/master/_images/2025-01-28-goaling-tb.png)

## [Thoughts](#thoughts)

Now the ingredients for Layered Learning is ready. Next post will cover how to combine these layers for an agent to perform complex <mark>playing</mark> behavior.