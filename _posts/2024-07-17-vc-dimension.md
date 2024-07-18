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

### Terms

- Given some finite set of data points \\(S = \({x}^{(1)}, .... , {x}^{(M)}\)\\) and some hypothesis \\(h \in H\\), applying \\(h\\) to each point in \\(S\\) results in a labelling
  - \\(\left(h\left(x^{(1)}\right), \cdots, h\left(x^{(M)}\right)\right)\\) is a vector of \\(M\\) +1 and -1. (Binary Classifier)
- Given \\(S\\), each hypothesis in \\(H\\) induces a labelling but **not necessarily a unique labelling**.
  - The set of labellings indcues by \\(H\\) on \\(S\\) is
    
    $$
    H(s)=\left\{\left(h\left(x^{(1)}\right), \cdots, h\left(x^{(M)}\right)\right) \mid h \in H\right\}
    $$
  
  - Note that even if \\(H\\) is infinite, set of labellings that you get from all of the hypothesis on \\(H\\) is **finite**.

### So what're we doing?

- In our previous learning theory bounds, where \\(H\\) was finite, the size of \\(H\\) stands for a measure of how complex or how rich and expressive the model is.

- Here, where \\(H\\) is infinite, we need slightly different measure for that, and we cen define this in terms of **how many possible labellings** we can generate on some dataset \\(S\\).

### Growth Function

Growth function of \\(H\\) is the maximum number of disinct labellings \\(H\\) can induce on *any* set of *M* data points:

$$
g_{\mathcal{H}}(M)=\max _{s:|s|=M}|\mathcal{H}(s)|
$$

- Larger the growth function is, more danger my model to be over-fitted.
- In binary classification task, \\(g_{\mathcal{H}}(M)\\) is always bounded to \\(2^M\\):

  $$
  g_{\mathcal{H}}(M) \leq 2^M \ \forall \ \mathcal{H} \ \text { and } \ M
  $$

- This has its own term, which is a **shatter**.
  - \\(H\\) *shatters* \\(S\\) if \\(\|\mathcal{H}(s)\| = 2^{M}\\)

  $$
  \text { If } \ \exists \ S \ \text { s.t. } \|S\|=M \text { and } \mathcal{H} \text { shatters } S \text {, then } g_{\mathcal{H}}(M)=2^M
  $$

### Exercise

1. \\(x^{(m)} \in \mathbb{R}^2\\) and \\(H =\\) all 2-dimensional linear separators, what is \\(g_{\mathcal{H}}(3)\\)?
  - We can find arrangements of m that makes \\(g_{H}\(3\) = 6 \\) (ex. when three points are arranged straight in a row). However, since we should consider all possible arrangement of m data points, \\(g_{H}\(3\) = 8\\).
  - This means this 2-dimensional linear separator \\(H\\) shatters \\(S\\).
    
2. \\(x^{(m)} \in \mathbb{R}^2\\) and \\(H =\\) all 2-dimensional linear separators, what is \\(g_{\mathcal{H}}(4)\\)?
  - \\(g_{H}\(4\) = 8 \leq 2^4\\).
  - This means no set of 4 points in linear decision boundaries in 2-dimensional can shatter.

## [VC Bound](#vc-bound)

In finite, realizable case: for any hypothesis set \\(M\\) and distribution \\(p*\\), if the number of labelled training data points satisfies:

$$
M \geq \frac{2}{\varepsilon}\left(\log _2\left(2 g_{M}(2 M)\right)+\log _2\left(\frac{1}{\delta}\right)\right)
$$

then with probability at least \\(1-\delta\\), all \\(h \in H\\) with \\(R(h) \geq \varepsilon\\) have \\(\hat{R}(h) > 0\\).

- This Vapnik-Chervonenkis(VC)-Bound theorem works for where our hypothesis set is infinite in size but we know that \\(c*\\) lives within that inifinite size hypothesis \\(H\\).
- However, there's diffult-to-solve problem here: the term \\(M\\) appears both sides of enequality equation.
- To walk around this issue, we introduce the concept of VC-Dimension.

## [VC Dimension](#vc-dim)
## [Revisit VC Bound](#revisit)
## [Trade Off in ML](#tradeoff)

---
{: data-content="footnotes"}

[^1]: Content and explanation originally *[from this video](https://youtu.be/fTWm2S5tFCo?si=wL9cLp_45FGRwic6), SanITtips*
