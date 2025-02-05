---
layout: post
category: RL
---

In my last post, I suggested stratrgy of realizing complex behavior by layering simple pre-trained reinforcement models. For recap the layer we utilized was:

	1.	Wandering
	2.	Painting
	3.	Moving to the goal

In this post, I’ll discuss the resulting agent that was realized using this layered method.

## Table of contents

- [Structure](#structure)
- [Result](#result)
- [Expandability](#expandability)

## [Structure](#structure)

This is how layered architecture works in realizing complex game-playing behavior. The agent’s behavior is constructed using a class called HierarchicalAgent. This agent performs actions based on the observations it receives from the environment and determines its next action through its internal composite_policy. The composite_policy first checks whether the agent’s color matches the color of the goal tile. If not, it calls the painting model to plan the sequence in which the agent should visit different locations. After this decision, the agent calls the goaling model, switching its goal to the color chips instead of the goal tile as an interim objective it must achieve.

### Pseudocode

```
HierarchicalAgent ← painting_model, goaling_model, wandering_model
HierarchicalAgent performs composite_policy ← observation from environment
	(1) check color
		if self_color != goal_color:
			call painting_model
		else:
			call goaling_model
	decision = action from painting_model
	(2) call goaling_model # goal is target color chip
```

## [Result](#result)

This is the result of layered agent performing on the Paint Your Partner environment. Compared to the chunked version of RL model, this approach effectively performed game actions without eccessive training time.

<video width="360" height="360" controls>
  <source src="https://raw.githubusercontent.com/nik-pitts/machinelearning601/master/_images/2025-02-04-layered-agent-result.mp4" type="video/mp4">
</video>

### An Issue

However, I also found an issue with this approach. Since the agent was simply performing the *goaling* action when making decisions, it couldn’t avoid color chips that would change its color and prevent it from achieving the target goal color. As a result, it fell into a loop of <mark>complete confusion</mark> when it should have been evading the color chips while progressing toward its final goal—matching the target color.

## [Expandability](#expandability)

However, this approach has a significant advantage— the agent’s actions scale to different environments. <mark>Expandability</mark> is one of the biggest obstacles in Reinforcement Learning because small changes in the environment can disrupt all prior learning, making the agent vulnerable to new conditions. However, the layered approach remains effective even with slight changes in the environment. This highlights the importance of a well-designed environment in addressing AI agents’ scaling challenges.

Eventhough each behavior was trained in 5x5 grid environment initially, agents could still perform well in 15x15 grid. Below video shows the agent playing the game in 10x10 environment.

<video width="500" height="500" controls>
  <source src="https://raw.githubusercontent.com/nik-pitts/machinelearning601/master/_images/2025-02-04-layered-agent-expandability.mp4" type="video/mp4">
</video>