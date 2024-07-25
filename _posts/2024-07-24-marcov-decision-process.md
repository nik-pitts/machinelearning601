---
layout: post
category: RL
---

This post covers **Marcov Decision Process** in detail, which forms a foundation of Reinforcement Learning.

## Table of contents

- [Problem Formulation](#problem-formulation)
- [Technical Definition of MDP](technical-definition-of-MDP)
- [3 Key Challenges of Reinforcment Learning](3-key-challenges-of-reinforcement-learning)
- [RL Objective Function](rl-objective-function)

## [Problem Formulation](#problem-formulation)

### Terms

- State Space \\(S\\)
- Action Space \\(A\\)
- Reward Function
  There are two different reward functions that leads to different definitions of the problem at a high-level, but these two functions do the same following thing: Reward function tells you how good some actions is in some state.
  - Stochastic, \\(p(r\|s, a)
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
  
## [Technical Definition of MDP](technical-definition-of-MDP)

In the context of reinforcement learning, we need to define what's known as ***Data Generation Process**. The idea is reinforcement learning algorithms, they're going to require some sort of training data to learn from, but *where does that training data come from?*. 

1. Start in some initial state \\(s_0\\)
2. For time step \\(t\\):
   1. Agent observes state \\(s_t\\)
   2. Agent takes action \\(a_t = \pi(s_t)\\)
      Based on some policy, not necessarily the optimal policy based on some mechanism, it takes an action in that state.
   3. Agent recieves reward \\(r_t \tilde p(r \| s_t, a_t)\\)
      Recives some reward, here it is written out stochastically, but this could be deterministic.
   4. Agent transitions to state \\(s_{t+1} \tilde p(s' \| s_t, a_t)\\)
      Ends up in some new state. \
   → These 2.1 ~ 2.4 creates training data set and are observation in my training data set. Also called as **observation tuple**. \
   → We repeat this until we hit some terminal state or until we reach some time horizon.
3. Total reward: \\(\sum_{t=0}^{\infty} \gamma{^t} \cdot r_t\\)
   * In the case of infinite time horizon reward.
   - \\(\gamma\\) = Discount Factor, \\(\gamma \in (0,1) \\)

## [3 Key Challenges of Reinforcment Learning](3-key-challenges-of-reinforcement-learning)

1. The algorithm has to gather its own training data.
   You need to find some way of gathering this training data such that you are **able to learn a reasonable policy** from your own training data set.
2. The outcome of taking some action is often **stochastic or unknown** until after the fact.
   - Room for error that might lead us down sub-optimal paths.
3. Decisions can have a delayed effect on future outcomes. (Exploration-Exploitation tradeoff)
   - Potential cost in terms of maximizing its **current** reward.
   - The system itself has some incentive to try and chase high reward actions, but that might lead to sub-optimal learning of the system.
   - Multi-armed Bandit Problem

### Multi-armed Bandit problem(Slot machine problem)

![multi-armed-bandit-problem](https://thumb.ac-illust.com/5f/5f65d7975caf523ce80ed30f340bfac1_t.jpeg)
*Multi-armed Bandit Problem Illustration*[^1]

- \\(\|S\| = 1\\), agent standing infront of slot machines.
- \\(A = \{1,2,3\}\\) pull slot machine 1, 2 or 3.
- Deterministic transition
- Rewards are stochastic and unknown to agent

In this context, the agent is going to figure out:
1. What those rewards distributions are
2. but also what is the sort of the **right policy** given that distribution over the reward is.

> As point 1 and 2 describe, it is not an easy problem to learn optimal solutions while dynamics of the system is *unknown*.

## [RL Objective Function](rl-objective-function)

- Objective: Find a policy \\(\pi^*=\underset{\pi}{\arg \max } \ V^\pi(s) \ \forall \ s \in S\\)
  - \\(\pi^*\\), optimal policy
- \\(V^\pi(s) = E\[\text{discounted total reward of starting in state} s \text{and executing policy} \pi \text{forever}\]\\)
  - The value of being in some state \\(s\\) conditioned on, or subject to this policiy \\(\pi\\) is expected discounted total reward of starting in this state \\(s\\) and just following the policy \\(\pi\\) tells you to do in each state.

---
{: data-content="footnotes"}

[^1]: Figure from *[this webpage](https://en.ac-illust.com/clip-art/1800887/isometric-projection-of-multiple-blue-slot-machines)*, MoanaAkasso