# Linear regression


Problem definition

**m** examples, **n** features

$$\theta = [\begin{matrix}
\theta_0^1, \theta_0^2,..\theta_0^m \\
\theta_1^1, \theta_1^2,..\theta_1^m \\

\theta_n^1, \theta_n^2,..\theta_n^m
\end{matrix}] - weights
$$

$$X = [\begin{matrix}
1, 1,..1\\
x_1^1, x_1^2,..x_1^m \\
x_n^1, x_n^2,..x_n^m
\end{matrix}] - features
$$

$$y = [\begin{matrix}
y^1\\
..\\
y^m
\end{matrix}] - target
$$
Linear regression approximation:
$$\overline{y} = h_\theta(X) = \theta^T*X$$

Cost function - Euclidian distance
$$
J(\theta) = \dfrac{1}{2m}\sum_{i=1}^m(h_\theta(X)-y)^2
$$

Gradient desent (simultaneously for every i)
$$\theta_i = \theta_i - \dfrac{\partial}{\partial \theta_i}J(\theta)  $$
