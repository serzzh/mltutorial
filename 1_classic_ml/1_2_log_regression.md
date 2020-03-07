# Logistic regression


Problem definition

Classification

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
\end{matrix}] ,\space
y^i \in \{0,1\} - target
$$
Logistic regression hypothesis representation:
$$\overline{y} = h_\theta(X) = g(\theta^TX) $$

$$g(z)=\dfrac{1}{1+e^{-z}}- sigmoid\space function$$

Interpretation of hypothesis output
 $$h_\theta(x) = p(y=1|x;\theta)$$

Cost function
$$
J(\theta) = -\dfrac{1}{m}\sum_{i=1}^m[y^i\log{h_\theta(x^i)}+(1-y^i)\log{(1-h_\theta(x^i))}]
$$

Gradient Descent (simultaneously for every i)
$$\theta_j = \theta_j - \alpha\dfrac{\partial J(\theta)}{\partial \theta_j}$$


$$\theta_j =  \theta_j - \dfrac{\alpha}{m}\sum_{i=1}^m{(h_\theta(x^i)-y^i)x_j^i}$$

Regularization term

$$
J(\theta) = -\dfrac{1}{m}\sum_{i=1}^m[y^i\log{h_\theta(x^i)}+(1-y^i)\log{(1-h_\theta(x^i))}] + \dfrac{\lambda}{2}\sum_{j=1}^n\theta_j^2
$$

$$\theta_j =  \theta_j(1-\alpha\dfrac{\lambda}{m}) - \alpha\dfrac{1}{m}\sum_{i=1}^m{(h_\theta(x^i)-y^i){x_j^i}}$$

Bonus - Gradient Descent derivation

$$
J(\theta) = -\dfrac{1}{m}\sum_{i=1}^m[y^i\log{h_\theta(x^i)}+(1-y^i)\log{(1-h_\theta(x^i))}]
$$
$$h_\theta(x)=\dfrac{1}{1+e^{-\theta^T x}}$$
$$
J(\theta) =\dfrac{1}{m}\sum_{i=1}^m[y^i\log{(1+e^{-\theta^T x^i})}+(1-y^i)(\theta^T x^i+\log{(1+e^{-\theta^T x^i}))}]
$$

$$
J(\theta) =\dfrac{1}{m}\sum_{i=1}^m[(1-y^i)\theta^T x^i+\log{(1+e^{-\theta^T x^i})}]
$$

$$ \dfrac{\partial J(\theta)}{\partial \theta_j} =  \dfrac{1}{m}\sum_{i=1}^m{(1-y^i-\dfrac{e^{-\theta^T x^i_j}}{1+e^{-\theta^T x^i_j}})x^i_j} =
\dfrac{1}{m}\sum_{i=1}^m{(h_\theta(x^i_j)-y^i)x_j^i}$$
$$\theta_j =  \theta_j - \dfrac{\alpha}{m}\sum_{i=1}^m{(h_\theta(x^i)-y^i)x_j^i}$$
