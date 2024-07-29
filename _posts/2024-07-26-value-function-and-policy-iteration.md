---
layout: post
category: RL
---

This post covers **Value Function and Policy Interation Algorithms** of Reinforcement Learning in deail. The post is written in reference to CMU 10401/601 taught by Henry Chai.

## Table of contents

- [Value Function](#value-function)
- [Optimality](#optimality)
- [Fixed Point Iteration](#fixed-point-iteration)

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
- \\(\mathbb{E}\left[R\left(s_t, \pi\left(s_t\right)\right)\right]\\): Reward received when taking action \\(\pi\(s_t\)\\) in state \\(s_t\\).
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
R(s, \pi(s)) + \gamma \sum_{s_1} p\left(s_1 \mid s, \pi(s)\right) \left[ R\left(s_1, \pi\left(s_1\right)\right) + \gamma \mathbb{E}\left[ R\left(s_2, \pi\left(s_2\right)\right) + \cdots \mid s_1 \right] \right]
$$

- \\(p\left(s_1 \mid s, \pi(s)\right)\\): Probability of transitioning **to state \\(s_1\\) from state \\(s\\)** when action \\(\pi(s)\\) is taken.
- The term inside the sum, \\(R\left(s_1, \pi\left(s_1\right)\right) + \gamma \mathbb{E}\left[R\left(s_2, \pi\left(s_2\right)\right) + \cdots \mid s_1\right]\\) represents the expected return starting from state \\(s_1\\) and following the policy \\(\pi\\).

> Transition Probability Notation: \\(p\left(s'|s,a\right)\\)
- \\(s'\\): **Current** state
- \\(a\\): **Action** taken in state \\(s\\)
- \\(s'\\): **Next** state

### Bellman Equations

$$
V^\pi(s)=R(s, \pi(s))+r \sum_{s_1 \in s} P\left(s_1 \mid s, \pi(s)\right) V^\pi\left(s_1\right)
$$

## [Optimality](#optimality)

What we care about is not the value of some arbitrary function, arbitrary policy \\(\pi\\). What we care about is the value function for the optimal poilicy \\(\pi*\\). Below is what that **value** looks like for the *optimal policy*:

- Optimal value function:

$$
V^*(s)=\max _{a \in A} R(s, a)+\gamma \sum_{s^{\prime} \in S} P\left(s^{\prime} \| s, a\right) V^*\left(s^{\prime}\right)
$$

\\(\Leftrightarrow\\) "Consider all of the possible actions I could take in every state, and find the one that maximizes the immediate reward plus the discounted future reward." This is again still a recursibe definition, I can define the optimal value of any state **in terms of the optimal value of all of the other states**. "Max" out front makes this a nonlinear systems of equations.

- Optimal policy:

$$
\pi^{*}(s)=\argmax _{a \in A} R(s, a)+\gamma \sum_{s^{\prime} \in S} P\left(s^{\prime} \| s, a\right) V^*\left(s^{\prime}\right)
$$

Once I know what the optimal value in every state is, I just take the action that achieves that optimal value and I take the \\(\argmax\\) in every state \\(\s\\).

Key Intuition: I can solfe for the optimal policy by solving the optimal value function. I can solve the optimal value function if I can solve this arbitrary nonlinear system of size of \\(s\\) equations and size of \\(s\\) variables. Note that we've been thinking a lot about *optimization* in the class, but this is not technically an optimization algorithm. We're not optimizing any objective function here, rather we're finding a solution to this system of equations. So what're the algorithms for solving this equation?

## [Fixed Point Iteration](#fixed-point-iteration)[^1]

### How It Works[^2]

- Given \\(f(x) = 0\\) write \\(x\\) in terms of \\(x = \dots\\)
- Label left side as \\(x_{n+1}\\) and right side with \\(x_n\\)
- Pick \\(x_1\\) and plug into equation
- Repeat until converges

### Example

$$
x^2 - x - 1 = 0
$$

1. Set equation \\(x = \tex{something}\\)
   $$
   x_{n+1}=1+\frac{1}{x_n}
   $$
2. Pick \\(x_1 = 2\\)
   $$
   x_2 = 1 + \frac{1}{2} = 1.5
   $$
3. Repeat \\(x_3\\), \\(x_4\\), \\(\dots\\)
   $$
   x_3 = 1 + \frac{1}{1.5} = 1.6666
   x_4 = 1 + \frac{1}{1.6666} = 1.6
   x_5 = 1 + \frac{1}{1.6} = 1.625
   x_6 = 1 + \frac{1}{1.625} = 1.612538462
   $$
4. Converging to \\(1.618 \dots\\)

> However, the same equation might not converge with different equation manipulation and initializing. If we set \\(x_{n+1} = \frac{1}{x_{n}-1}\\) and \\(x_1\\) to 1.6, for example, it will not converge.

#### When Converge?

- When expressing \\(f(x)=0\\) as \\(x=g(x)\\), choose such that \\(\|g'(x)\| < 1\\) \text{at} x = x_o \text{where} x_o\\) is some initial guess called *fixed point iterative scheme*.[^3]
- The fact that the discount factor is strictly less than one means that in the reinforcement setting, fixed point iteration will converge to the optimal value function. 
  
---
{: data-content="footnotes"}

[^1]: Reference *[this video](https://youtu.be/OLqdJMjzib8?si=Pw0xD966jp1S3cKr)*, Fixed Point Iteration, Oscar Veliz
[^2]: Same Video, 0:32
[^3]: Reference *[this page](https://byjus.com/maths/fixed-point-iteration/#:~:text=The%20fixed%20point%20iteration%20method%20uses%20the%20concept%20of%20a,g(x)%20%3D%20x.)*, byjus.com


