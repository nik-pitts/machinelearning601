---
layout: post
category: RL
---

This post covers **Value Function and Policy Interation Algorithms** of Reinforcement Learning in deail. The post is written in reference to CMU 10401/601 taught by Henry Chai.

## Table of contents

- [Value Function](#value-function)

## [Value Function](#value-function)

Recall the concept of value function below:

\\(V^\pi(s)=\mathbb{E}\\) [discounted total reward of starting in state \\(s\\) and executing policy \\(\pi\\) forever]

In a more precise and quantitative manner, we can re-write our value function:

$$
\begin{align}
V^\pi(s) & = \mathbb{E}\left[R\left(s_0, \pi\left(s_0\right)\right)+\gamma R\left(s_1, \pi\left(s_1\right)\right)+\gamma^2 R\left(s_2, \pi\left(s_2\right)\right)+\cdots \mid s_0=s\right] \\
&=R(s, \pi(s))+\gamma \mathbb{E}\left[R\left(s_1, \pi\left(s_1\right)\right)+\gamma R\left(s_2, \pi\left(s_2\right)\right)+\ldots \mid s_0=s\right] \\
&=R(s, \pi(s))+\gamma \sum_{s_1 \in s} p\left(s_1 \mid s, \pi(s)\right)\left(R\left(s_1, \pi\left(s_1\right)\right)\right.
\left.\quad+\gamma \mathbb{E}\left[R\left(s_2, \pi\left(s_2\right)\right)+\cdots \mid s_1\right]\right)
\end{align}
$$

### Terms

- State Space \\(S\\)
- Action Space \\(A\\)
- Reward Function
  There are two different reward functions that leads to different definitions of the problem at a high-level, but these two functions do the same following thing: Reward function tells you how good some actions is in some state.
  - Stochastic, \\(p(r\|s, a)\\)
    Returns *distribution* over rewards given some state and action.
  - Deterministic, \\(R: S \times A \rightarrow \mathbb{R}\\)
- Transition Function
  Every action I take brings me to a new state that can again either be a deterministic function \\(f\\) or some distribution over next states.
  - Stochastic, \\(p(s'\|s,a)\\)
  - Deterministic, \\(\delta: S \times A \rightarrow S\\)
- Policy \\(\pi: S \rightarrow A\\)
  Specifies an action to take in every state. This that we are trying to learn.
- Value Function \\(V^{\pi}: S \rightarrow \mathbb{R}\\)
  - Some measure of how good the *policy* \\(\pi\\) is, given an initial state \\(S\\)
  - Optimal policy maximizes value function in every state.

Our goal is to define our value function in a recursive way.
  
---
{: data-content="footnotes"}

[^1]: Figure from *[this webpage](https://en.ac-illust.com/clip-art/1800887/isometric-projection-of-multiple-blue-slot-machines)*, MoanaAkasso
