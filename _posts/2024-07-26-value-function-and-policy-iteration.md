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
V^\pi(s) &= \mathbb{E}\left[R\left(s_0, \pi\left(s_0\right)\right) + \gamma R\left(s_1, \pi\left(s_1\right)\right) + \gamma^2 R\left(s_2, \pi\left(s_2\right)\right) + \cdots \mid s_0 = s\right] \\
&= R(s, \pi(s)) + \gamma \mathbb{E}\left[R\left(s_1, \pi\left(s_1\right)\right) + \gamma R\left(s_2, \pi\left(s_2\right)\right) + \cdots \mid s_0 = s\right] \\
&= R(s, \pi(s)) + \gamma \sum_{s_1} p\left(s_1 \mid s, \pi(s)\right) \left( R\left(s_1, \pi\left(s_1\right)\right) + \gamma \mathbb{E}\left[ R\left(s_2, \pi\left(s_2\right)\right) + \cdots \mid s_1 \right] \right)
\end{align}
$$

### Step by Step Break Down

1. Definition of Value Function

$$
V^\pi(s) = \mathbb{E}\left[R\left(s_0, \pi\left(s_0\right)\right) + \gamma R\left(s_1, \pi\left(s_1\right)\right) + \gamma^2 R\left(s_2, \pi\left(s_2\right)\right) + \cdots \mid s_0 = s\right]
$$

- \\(V^\pi(s)\\): Expected sum of rewards
- \\(\mathbb{E}\left[R\left(s_t, \pi\left(s_t\right)\right])\\): Reward received when taking action \\(\pi\(s_t\)\\) in state \\(s_t\\).
- \\(\gamma\\): Discount factor

2. Recursive Formulation

$$
R(s, \pi(s))+\gamma \mathbb{E}\left[R\left(s_1, \pi\left(s_1\right)\right)+\gamma R\left(s_2, \pi\left(s_2\right)\right)+\ldots \mid s_0=s\right]
$$

This equation expresses the value function recursively. The value of starting in state \\(s\\) and following policy \\(\pi\\) is the *immediate reward* **plus** the discounted expected value of the *subsequent states*.

- \\(R(s, \pi(s))\\): Immediate reward for taking action \\(\pi(s)\\) in state \\(s\\).
- The second term represents the discounted expected value of future rewards starting from state \\(s_1\\).

3. Incorporating State Transition Probabilities

$$
R(s, \pi(s))+\gamma \sum_{s_1 \in s} p\left(s_1 \mid s, \pi(s)\right)\right[R\left(s_1, \pi\left(s_1\right)\right)\right\left +\gamma \mathbb{E}\right[R\left(s_2, \pi\left(s_2\right)\right)+\cdots \mid s_1\]\]
$$

- p\left(s_1 \mid s, \pi(s)\right): Probability of transitioning **to state \\(s_1\\) from state \\(s\\)** when action \\(\pi(s)\\) is taken.
- The term inside the sum, \\(R\left(s_1, \pi\left(s_1\right)\right)\right\left +\gamma \mathbb{E}\[R\left(s_2, \pi\left(s_2\right)\right)+\cdots \mid s_1\]\\) represents the expected return starting from state \\(s_1\\) and following the policy \\(\pi\\).

> Transition Probability Notation \
$$
p(s' \| s, a)
$$
- \\(s'\\): **Current** state
- \\(a\\): **Action** taken in state \\(s\\)
- \\(s'\\): **Next** state

  
---
{: data-content="footnotes"}

[^1]: Figure from *[this webpage](https://en.ac-illust.com/clip-art/1800887/isometric-projection-of-multiple-blue-slot-machines)*, MoanaAkasso
