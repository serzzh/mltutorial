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
$$\overline{y} = h_\theta(X) = \theta^T\times{X}$$

Cost function - Euclidian distance
$$
J(\theta) = \dfrac{1}{2m}\sum_{i=1}^m(h_\theta(x^i)-y^i)^2
$$

Gradient desent (simultaneously for every i)
$$\theta_j = \theta_j - \alpha\dfrac{\partial J(\theta)}{\partial \theta_j}$$
$$ \dfrac{\partial J(\theta)}{\partial \theta_j} =  \dfrac{2}{2m}\sum_{i=1}^m{(h_\theta(x^i)-y^i)\times{\dfrac{\partial}{\partial \theta_j}(h_\theta(x^i)-y^i)}} =
\dfrac{1}{m}\sum_{i=1}^m{(h_\theta(x^i)-y^i)\times{x_j^i}}$$
$$\theta_j =  \theta_j - \dfrac{\alpha}{m}\sum_{i=1}^m{(h_\theta(x^i)-y^i)\times{x_j^i}}$$

Regularization term

$$
J(\theta) = \dfrac{1}{2m}[\sum_{i=1}^m(h_\theta(x^i)-y^i)^2 + \lambda\sum_{j=1}^n\theta_j^2]
$$

$$\theta_j =  \theta_j(1-\alpha\dfrac{\lambda}{m}) - \alpha\dfrac{1}{m}\sum_{i=1}^m{(h_\theta(x^i)-y^i){x_j^i}}$$

Bonus - normal equation

$$J(\theta) = \dfrac{1}{2m}\sum_{i=1}^m(h_\theta(x^i)-y^i)^2 = (X\theta -y)^T(X\theta -y)$$
$$\dfrac{\partial J(\theta)}{\partial \theta}=\dfrac{1}{m}(X^TX\theta-X^Ty)=0$$


$$\theta = (X^TX)^{-1}X^Ty$$
