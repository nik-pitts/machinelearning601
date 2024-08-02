---
layout: post
category: RL
---

Reinforcement Learning covers whole new concept compared to traditional supervised learning. Therefore, the idea of RL is difficult to grasp in a first sense. While studying RL, I realized obtaining concrete understanding of basic building blocks that consists RL is a highly crucial task. This post covers second crucial stepping stone in understanding reinforcement learning, **Bellman Equation**. Again, the post was written in reference to awesome introductory video about reinforcement learning made by the Youtube channel *Mutual Information*. Detailed information is written below footnote.[^1]

## Table of contents

- [Dynamic Programming](#dynamic-programming)

## [Dynamic Programming](#dynamic-programming)

### What is Dynamic Programming?

Dynamic Programming (DP) is a method used in mathematics and computer science to solve complex problems by breaking them down into simpler subproblems.  By solving each subproblem only once and storing the results, it avoids redundant computations, leading to more efficient solutions for a wide range of problems.[^2] Key notions involved in dynamic programming are **optimal decesion making** and **solving subproblems**.

### Condtions to Become a DP Problem

![DP Problems](https://favtutor.com/resources/images/uploads/blobid0.png)
*What makes DP Problems*[^3]

- Optimal Substructures
  In computer science, a problem is siad to have optimal substructure if an optimal solution can be constructed from optimal solutions of its subproblems.[^5]

- Overlapping Subproblems
  In computer science, a problem is said to have overlapping subproblems if the problem can be broken down into subproblems which are reused several times or a recursive algorithm for the problem sovles the same subproblem over and over rather than always generating new subproblems. For exmaple, computing the Fibonacci Sequence exhibits overlapping subproblems.[^4]

## [Bellman Equations](#bellman-equation)

### Bellman Equation for \\(\mathcal{v}_{\pi}(s)\\)


---
{: data-content="footnotes"}

[^1]: Referenced *[this video](https://youtu.be/NFo9v_yKQXA?si=j2BCf36NgJYOfF2K)*, Mutual Information, Reinforcement Learning, by the Book
[^2]: *[Definition of Dynamic Programming](https://www.geeksforgeeks.org/dynamic-programming/)* ,Geeks For Geeks
[^3]: Image from *[this page](https://favtutor.com/resources/images/uploads/blobid0.png)*, FavTutor, Shivali Bhadaniya, Dynamic Programming in Python: Top 10 Problems (with code)
[^4]: Definition form *[Wiki page](https://en.wikipedia.org/wiki/Optimal_substructure#:~:text=In%20computer%20science%2C%20a%20problem,greedy%20algorithms%20for%20a%20problem.)*
[^5]: Definition form *[Wiki page](https://en.wikipedia.org/wiki/Overlapping_subproblems#:~:text=In%20computer%20science%2C%20a%20problem,than%20always%20generating%20new%20subproblems.)*
