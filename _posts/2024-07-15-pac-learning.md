---
layout: post
category: Theory
---

This post covers **Probably Approbimately Correct** learning theory.
The content was written in reference to course content from *Henry Chai(CMU 10-401/601)* and [this video](https://youtu.be/fTWm2S5tFCo?si=wL9cLp_45FGRwic6), *Probably Approximately Correct(PAC) Learning (KTU CS467 Machine Learning Module 2)*.

## Table of contents
- [Statistical Learning Theory Model](#stat-ltm)
- [Types of Error](#error)
- [Types of Risk](#risk)
- [Our interest](#interest)
- [Rectangle PAC](##rec-pac)

## [Statistical Learning Theory Model](#stat-ltm)

Statistically, we can precisely define our model as below:

1. Data points are generated from some *unknown* **distribution**.
   $$ x^(n) ~ p*(x) $$
2. Labels are generated from some *unknown* function.
   $$ y^(n) = c*(x^(n)) $$
   - c*: target generating function
3. Learning algorithm chooses the hypothsis(or classifier) with **lowest tarining error rate** from a specified *hypotheisis(classifier)* set, \$H\$.
4. But! Our goal is to return a hypothesis(or classifier) with **low true error rate**. This means we expect good generalisation.

## [Types of Error](#error)

Let's review types of error we have been through.

- True error rate:
  - This is the thing we *ultimately* care about.
  - How well your hypothesis will perform *on average* across **all possible data points**.
  - However, we don't know this. Therefore, we *approximate* true error rate by introducing different types of error rate as below.

- Test error rate:
  - Used to evaluate hypothesis performance with data set that wasn't used to turn hyperparameters or training process.
  - This is a reasonable estimate of our hypothesis's true error.
 
- Validation error rate
  - Used to set hypothesis hyperparameters.

- Training error rate
  - Used to set model parameters.
  - Very *optimistic* estimate of hypothesis's true error.
 
* Note: in PAC learning we focus on relationship between true error rate and training error rate.

## [Types of Risk](#risk)

Terminology alert! Although conceptuall we're using same notion of true error rate and training error rate, we are going to use term *expected risk* and *empirical risk* in this section. Let's think error rate as **risk** that your model makes mistakes.

- Expected risk of hypothesis $h$ (true error)
  $$R(h) = P_x~p*(c*(x) \ne h(x))$$
  - Sampling over **all possible** data points under the distribution $p*$.

- Empirical risk of hypothesis $h$ (training error)
  $$\begin{aligned}
    \hat{R}(h) & =P_{\boldsymbol{x} \sim \mathcal{D}}\left(c^*(\boldsymbol{x}) \neq h(\boldsymbol{x})\right) \\
    & =\frac{1}{N} \sum_{n=1}^N \mathbb{1}\left(c^*\left(\boldsymbol{x}^{(n)}\right) \neq h\left(\boldsymbol{x}^{(n)}\right)\right) \\
    & =\frac{1}{N} \sum_{n=1}^N \mathbb{1}\left(y^{(n)} \neq h\left(\boldsymbol{x}^{(n)}\right)\right)
    \end{aligned}
  $$
  - Sampling over **training dataset D**.
 
Note that \$\mathcal{D}=\left\{\left(\boldsymbol{x}^{(n)}, y^{(n)}\right)\right\}_{n=1}^N\$ is the training data set and $x ~ D$ denotes a point sampled uniformly at random from \$D\$.

## [Our Interest](##interest)

To sum up, let's recall what our interests are when we are building a learning model. Withour a doubt, we are interested in the **true function**, \$c*\$. However, we have no idea what it is, so we try to approximate it. And this goal can be rewritten in format such that:

- Expected risk minimizer
  $$h^*=\underset{h \in \mathcal{H}}{\operatorname{argmin}} R(h)$$
  - The lowest "true" error rate

- Empirical risk minimizer
  $$\hat{h}=\underset{h \in \mathcal{H}}{\operatorname{argmin}} \hat{R}(h)$$
  - The lowerst "training" error rate
 
This observation concludes into one question: **Given a hypothesis with zero/low *"training error"*, what can we say about its true error?** Can training error give some insight about true error and true performance of our model?

## [Rectangle PAC](##rec-pac)


---
{: data-content="footnotes"}

[^1]: Figure source *[from here](https://github.com/Coding-Lane/Training-Word-Embeddings---Scratch), Coding Lane*
