---
layout: post
category: Theory
---

This post covers **VC Dimension** learning theory.
The content was written in reference to course content from *Henry Chai(CMU 10-401/601)*.

## Table of contents
- [Recall PAC Learning](#recall)
- [Intuition](#intuition)
- [Growth Function](#growth-function)
- [VC Bound](#vc-bound)
- [VC Dimension](#vc-dim)
- [Revisit VC Bound](#revisit)
- [Trade Off in ML](#tradeoff)

## [Recall PAC Learning](#recall)

In PAC learning, we mathematically verified how many sample size of \\(m\\) we need to ensure our hypothesis \\(h\\) of at least \\(1-\delta\\) confidency and to have an accuracy of \\(\varepsilon\\).

$$
m >\dfrac{4}{\varepsilon }\ln \dfrac{4}{\delta }
$$

To restate, for a finite hypothesis set \\(H\\) and arbitrary distribution \\(p*\\), given a training data set \\(S\\) s.t. \\(\|S\| = M \\), all \\(h \in H\\) have:

$$
R\left( h\right) \leq \widehat{R}\left( h\right) +\sqrt{\dfrac{1}{2M}\left( \ln \left( |H|\right) +\ln \left( \dfrac{2}{\delta }\right) \right) }
$$

with probability at least \\(1-\delta\\).

However, there was an assumption we made to derive this. That is, we conveniently set our hypothesis set \\(H\\) to be *finite*. But, what if our hypothesis set is *infinite*, which sounds more common case? Here, the notion of VC Dimension comes in to pull out working insight even for the case of our hypothesis set is infinite.

## [Intuition](#intuition)

The intuition behind VC bounds is this question:

If two hypothesis \\(h_1, h_2 \in H\\) are very **similar**, then the events

- \\(h1\\) is consistent with the first \\(m\\) training points.
- Like wise, \\(h2\\) is consistent with the first \\(m\\) training points.

In other words, labelling result among hypothesis will **overlap** a lot!

> Some notion of measuring how similar two hypothesis or two classifiers are comparing how each of them would label the points on a same data set.

## [Growth Function](#growth-function)

Terminology alert! Although conceptuall we're using same notion of true error rate and training error rate, we are going to use term *expected risk* and *empirical risk* in this section. Let's think error rate as **risk** that your model makes mistakes.

- Expected risk of hypothesis \\(h\\) (true error)
  $$R(h) = P_x \sim p*(c*(x) \ne h(x))$$
  - Sampling over **all possible** data points under the distribution \\(p*\\).

- Empirical risk of hypothesis $h$ (training error)
  $$\begin{aligned}
    \hat{R}(h) & =P_{\boldsymbol{x} \sim \mathcal{D}}\left(c^*(\boldsymbol{x}) \neq h(\boldsymbol{x})\right) \\
    & =\frac{1}{N} \sum_{n=1}^N \mathbb{1}\left(c^*\left(\boldsymbol{x}^{(n)}\right) \neq h\left(\boldsymbol{x}^{(n)}\right)\right) \\
    & =\frac{1}{N} \sum_{n=1}^N \mathbb{1}\left(y^{(n)} \neq h\left(\boldsymbol{x}^{(n)}\right)\right)
    \end{aligned}
  $$
  - Sampling over **training dataset D**.
 
Note that \\({D}=\left\lbrace\left(\boldsymbol{x}^{(n)}, y^{(n)}\right)\right\rbrace_{n=1}^N\\) is the training data set and \\(x \sim D\\) denotes a point sampled uniformly at random from \\(D\\).

## [VC Bound](#vc-bound)

To sum up, let's recall what our interests are when we are building a learning model. Withour a doubt, we are interested in the **true function**, \\(c*H\\). However, we have no idea what it is, so we try to approximate it. And this goal can be rewritten in format such that:

- Expected risk minimizer
  $$h^*=\underset{h \in \mathcal{H}}{\operatorname{argmin}} R(h)$$
  - The lowest "true" error rate

- Empirical risk minimizer
  $$\hat{h}=\underset{h \in \mathcal{H}}{\operatorname{argmin}} \hat{R}(h)$$
  - The lowerst "training" error rate
 
This observation concludes into one question: **Given a hypothesis with zero/low *"training error"*, what can we say about its true error?** Can training error give some insight about true error and true performance of our model?

## [VC Dimension](#vc-dim)

### 1. PAC Criterion

$$
P(|R(h)-\hat{R}(h)| \leq \varepsilon) \quad \geq \quad 1-\delta \quad \forall \quad h \in \mathcal{H}
$$

> \\(R(h)\\): Empirical risk \
> \\(\hat{R}(h)\\): Expected risk \
> \\(\varepsilon\\): Difference between expected and empirical risk, **upper bound error**. \
> \\(\delta\\): Probability of failure. \
> \\(1-\delta\\): Confidency \
> \\(P(|R(h)-\hat{R}(h)| \leq \varepsilon)\\): Desirable property of a model

#### \\(\varepsilon\\)

Since for any set of training examples \\(T \subset X\\) there may be multiple hypothesis consistent with \\(T\\). However, the learner cannot be guaranteed to choose the one corresponding to the target concept, unless it is trained on every instance \\(X\\), which is unrealistic. **Therefore, we don't require L to output a zero error hypothesis - only require that error be bounded by a constant \\(\varepsilon\\) that can be made arbitrarily small**.

#### \\(\delta\\)

Since training examples drawn at random, there must be a non-zero probability that they'll be misleading. **Therefore, we don't want L to succeed for every randomly drawn sequence of training examples - only require that it's probability of failure be bounded by a constant \\(\delta\\) that can also be made arbitrarily small**.

### 2. PAC Learning Model: General Setting

- \\(x\\): Set of all possible instances over which target functions are to be defined.
- \\(C\\): Set of target concepts the learner may be asked to learn. Where each \\(c \in C\\), \\(C\\) may be viewed as a boolean-valued function \\(c:x \rightarrow \{0, 1\}\\). If \\(x\\) is a positive example \\(C(x)=1\\); If \\(x\\) is a negative example \\(C(x)=0\\).
- \\(D\\): Examples are drawn at random from \\(x\\) according to a probability distribution \\(D\\).
- \\(H\\): A learner \\(L\\) considers a set of hypothesis \\(H\\) and after observing some sequence of training examples, outputs a hypothesis \\(h \in H\\) which is its estimate of \\(C\\).
- The true error of hypothesis \\(h\\) (denoted \\(\operatorname{error}_D(h)\\)) w.r.t. target concept \\(c\\) and distribution \\(D\\) is the probability that \\(h\\) will misclassify an instance drawn at random according to \\(D\\).

$$ 
\operatorname{error}_D(h)=\operatorname{Pr}_{x \in D}[c(x) \neq h(a)] 
$$
  
### 3. Rectangle PAC Learning

![error-region](https://raw.githubusercontent.com/nik-pitts/machinelearning601/master/_images/2024-07-15-error-region.jpg)

- Our goal is to find the best \\(h\\), that approximate target function \\(c\\).
- \\(c\\): Real target function
- \\(h_s\\): The tightest possible rectangle around a set of positive training example.
- Since \\(h_s \subset c\\), error region = \\(c-h\\).
- **To be approximately correct, the area of error region should be bounded by \\(\varepsilon\\).**

> Idea: If at least **one** of the side was misclassified, all the samples of training set, we'll get bad hypothesis.

![strips](https://raw.githubusercontent.com/nik-pitts/machinelearning601/master/_images/2024-07-15-strips.jpg)

- Error region = Sum of four rectangular strips < \\(\varepsilon\\).
- Each strip is at most \\(\varepsilon / 4\\)
- Probability of a positive example falling in any of the strip(error region) = \\(\varepsilon / 4\\)
- Probability that a radomly drawn positive example misses a strip = \\(1-\varepsilon / 4\\)
- Probability of \\(m\\) examples miss a strip = \\((1-\varepsilon / 4)^m\\)
- Probability of \\(m\\) examples miss *any one of* strip \\(<4(1-\varepsilon / 4)^m\\)

To solve PAC learning,

$$
\begin{aligned}
& 4\left( 1-\varepsilon /4\right) ^{m} <\delta \quad \text{[Hypothesis PAC]} \\
& \text{Using inequality,} \; 1-x \leq e^{-x} \quad \text{[x = $\varepsilon$ / 4]} \\
& 4\left( 1-\varepsilon /4\right) ^{m}\leq 4e^{-m\varepsilon /4} <\delta \\
& m >\dfrac{4}{\varepsilon }\ln \dfrac{4}{\delta }
\end{aligned}
$$

### 4. Conclusion

If we want to have an accuracy of \\(\varepsilon\\) and confidence of at least \\(1-\delta\\), we have to choose a sample size \\(m\\) s.t

$$
m >\dfrac{4}{\varepsilon }\ln \dfrac{4}{\delta }
$$

## [Revisit VC Bound](#revisit)
## [Trade Off in ML](#tradeoff)

---
{: data-content="footnotes"}

[^1]: Content and explanation originally *[from this video](https://youtu.be/fTWm2S5tFCo?si=wL9cLp_45FGRwic6), SanITtips*
