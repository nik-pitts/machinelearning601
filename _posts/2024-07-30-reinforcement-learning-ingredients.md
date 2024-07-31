---
layout: post
category: RL
---

Reinforcement Learning covers whole new concept compared to traditional supervised learning. Therefore, the idea of RL is difficult to grasp in a first sense. While studying RL, I realized obtaining concrete understanding of basic building blocks that consists RL is a highly crucial task. This post covers very basic ingredients of RL in an extremely step by step approach. Also, the post was written in reference to awesome introductory video about reinforcement learning made by the Youtube channel 
*Mutual Information*. Detailed information is written below footnote.[^1]

## Table of contents

- [What the hack is MDP](#what-the-hack-is-mdp)
- [Ingredients of RL](#ingredients-of-rl)
- [Conceptual Sequence](#conceptual-sequence)
- [Our Ultimate Question](#our-ultimate-question)
- [Value Functions](#value-functions)

## [What the hack is MDP](#what-the-hack-is-mdp)

Let's start with MDP, Markov Decision Process. MDP is the term we firstly encounter when start to learn Reinforcement Learning. At least for me, the concept of MDP was far more easier when I study in a verbal form, not through an equation. To explain it verbally, **Finite Markov Decision Process** is a type of decision process that has a property of *Markov Property* with *finite sets of states, actions and rewards*. Then, what is the **Markov Property**?

### What is Markov Property?

In a nutshell, Markov Property means that the probability of *next state* and *rewards* only depends on **current state and action**, no matter the *history* of states and actions. That means, under Markov Property, you only concentrates on the present when making some decisions.

Now, since we understood what MDP is, let's build more on top of this.

## [Ingredients of RL](#ingredients-of-rl)

To concretely understand how reinforcement learning works, it is important to fully digest ingredients that are used in modeling environment, agent behaviour, and agent-environment interaction. Let's start with ingredients for the environment.

### Ingredients for the Environment

1.Timestep \\(t\\):
  $$
  t \in {0, 1, 2} \text{  \\timestep is a discrete value}
  $$
2. State \\(s\\):
  $$
  s \in \mathcal{S}
  $$
3. Action \\(a\\):
  $$
  a \in \mathcal{A(s)} \text{  \\set of actions that agent can take depending on which state agent is in}
  $$
4. Reward \\(r\\):
  $$
  r \in \mathcal{R} \in \mathbb{R} \text{  \\a real number in a finite subset of real number line}
  $$

### Ingredients for the Agent Behavior

To model agent's behaviour, we use a concept called **Policy**. The policy could be either stochastic or deterministic.

- Stochastic: \\(\pi(a\|s)\\)
  In a stochastic way, agent's behaviour is represented as a probability of doing some action \\(a\\) when it is in a particular state \\(s\\). Below are properties of policy.
  - If the state chages, then so do the action probability.
  - Agent select action by sampling randomly from the distribution.
  - **Policy changes overtime**
    This means policy affects *how data is collected*.

- Deterministic: \\(a = \pi(s)\\)
  If policy involves no randomness, then the **one** action could be taken from each state.

### Ingredients for the Agent - Environment Interaction

To model interactions between agent and environment, we use a function called **Transition Function**. In a nutshell, transition function captures *dynamics* of agent - environment interaction. In a formula we write transition function as below: 

$$
p(s',r | s, a) = p(S_t+1 = s', R_t+1=r | S_t=s, A_t=a)
$$

The function means **probability of next state and reward** being some values *given the current state and action to be taken*. This captures the concept of MDP, where future state and reward are with repect to *current* state and action.


## [Conceptual Sequence](#conceptual-sequence)

Then, let's look into how these individual ingredients are intangled with one another and could be interpreted in a sequential process of learning.

$$
\begin{align}
&s_0 \ \longrightarrow \ \pi(a|s_0) \ \longrightarrow \ p(s',r|s_0,a) \ \longrightarrow \ \text{takes stochastic action a} \longrightarrow \ \text{recieves} \ R_1 \ \text{reward and given} \ s_1 \text{ new state} \\
&s_1 \ \longrightarrow \ \pi(a|s_1) \ \longrightarrow \ p(s',r|s_1,a) \ \hookrightarrow \text{this continues} \dots
\end{align}
$$

Let's comprehend this sequence. To initiate the process, initial state \\(s_0\\) is given. Given this initial state, policy stochastically determines action to take. Given state \\(s_0\\) and action \\(a\\), transition kicks in. Then this transition function returns next sate \\(s_1\\) and reward \\(r_1\\) the agent recieves. We record this reward and the process keep unfolds feeding \\(s_1\\) as a new state.

## [Our Ultimate Question](#our-ultimate-question)

Okay, we've been through a lot. However, it's easy to get lost. Let's recall what was our purpose of this learning. What we've wanted as a result of this learning? We wanted the agent to learn the **optimal policy**. Then, how can we define optimal policy? We can do this by comparing return values of each policy, and in the context of reinforcement learning the return value would be **accumulation of total rewards**. We name this property as \\(G_t\\), meaning **sum of total future rewards**.

### \\(G_t\\): Sum of Total Future Rewards

$$
\begin{align}
&G_t = \sum^{T}_{k=t+1}\gamma^{k-t-1}R_k
&\Longleftrightarrow \ R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots + \gamma^{T-1} R_T
\end{align}
$$

The format of \\(G_t\\) enlights the possibility of forming this equation in a recursive way. Let's consider a case where \\(T=3 \text{ and } t=0, 1, 2\\) respectively.

i) t=0 \
$$
\begin{align}
G_0
& = \sum^{3}_{k=1}\gamma^{k-1}R_k \\
& = \gamma^0R_1 + \gamma^1R_2 + \gamma^2R_3
\end{align}
$$

ii) t=1 \
$$
\begin{align}
G_1
& = \sum^{3}_{k=2}\gamma^{k-2}R_k \\
& = \gamma^0R_2 + \gamma^1R_3
\end{align}
$$

iii) t=2 \
$$
\begin{align}
G_2
& = \sum^{3}_{k=3}\gamma^{k-3}R_k \\
& = \gamma^0R_3
\end{align}
$$

Note that \\(G_t\\) refers to the sum of **future** reward from current time step \\(t\\) to terminating time step \\(T\\). Also, note that future rewards are discounted by the discount factor \\(\gamma\\) gradually. In a way, we can explicitly express our goal as 

$$
\max_{\pi} \mathbb{E}_{\pi}[G_t]
$$

- Select policy that would maximize the **Expected Retrun**.
- In other words, pick a polciy that has the highest return value.

## [Value Functions](#value-functions)

So far so good. However, we have one more step left before wrapping up this section. To solve \\(G_t\\), there is a neccessity to break down this further. Inside \\(G_t\\), states and actions are jammed together. We need to disect them in order to compute expected returns. Here, I will introduce two new functions that do this work namely **State-Value Function** and **Action-Value Function**.

### State-Value Function: \\(\mathcal{V}_{\pi}(s)\\)

We define state-value function as below:

$$
\mathcal{V}_{\pi}(s) = \mathbb{E}_{\pi}\[G_t \| S_t=s\]
$$

State-value function takes state as an input, the subscribe \\(\pi\\) means that the function depends on policy \\(\pi\\), and the output \\(v\\) refers to some number, or value. Similar to \\(G_t\\) the output of this function is an expectation of the return given the agent at state \\(s\\), and uses policy \\(\pi\\) to determine its action. Ultimately, state-value function is what we're trying to optimize, but it's **broken w.r.t states**.

### Action-Value Function: \\(Q_{\pi}(s,a)\\)

Similarly, we define action-value function as below:

$$
Q_{\pi}(s,a) = \mathbb{E}_{\pi}[G_t \| S_t=s, A_t=a]
$$

The difference between state-value function and action-value function is that action-value function conditions on **actions** as well. To unfold the above equation, we can say, expected return given the agent at state \\(s\\), and **takes action \\(a\\)**. Likewise, action-value function is what we're trying to optimize, but it's **broken w.r.t states and actions**.

### Why we need these functions?

Recall our goal finding an optimal policy \\(\pi*\\). If we restate this optimal policy using value functions, it would be **"policy is optimal if it achieves highest possible expected return for all states"**. Therefore we are trying find optimal policy such that:

$$
\mathcal{V}_{\pi*}(S) \geq \mathcal{V}_{\pi}(S)
$$

How about action-value functions? The underlying notion is same as state-value function. However, this time we're conditioning state-action pairs istead of states only. In summary, **value functions provide instructions about how to behave optimally.**

### Summary of relationship between \\(G_t\\) and Value Functions

- \\(G_t\\): Provides the cumulative return from time step \\(t\\) onwards. It's a specific realization of rewards over a trajectory.
- \\(V(s)\\): Breaks down the return into expected value **given only the state \\(s\\) and the policy**. It's a generalization of the return considering all possible trajectories starting from state \\(s\\).
- \\(Q(s,a)\\): Breaks down the return into expected value given **both the state \\(s\\) and the action \\(a\\)**. It helps in understanding how good it is to take action \\(a\\) in state \\(s\\) and then follow the policy.

In essence, \\(V(s)\\) and \\(Q(s,a)\\) provide a more manageable way to evaluate and optimize policies by summarizing expected returns based on states and actions.

---
{: data-content="footnotes"}

[^1]: Referenced *[this video](https://youtu.be/NFo9v_yKQXA?si=j2BCf36NgJYOfF2K)*, Mutual Information, Reinforcement Learning, by the Book
