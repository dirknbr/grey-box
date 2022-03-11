# Semi linear model (grey box)

We have outcome y and inputs (Z, x). We suspect some nonlinear effects of Z on y
but don't care about the shape (black box). We do however care that x has a linear relationship with y. So f() will be a full neural network while g() will
be a linear function such as $a x^e$. We want y to be differentiable with respect to x (holding Z constant).   

$y = f(Z) + g(x)$

To restrict e to be in [0, 1] we use the sigmoid over raw e. 

In our simulation all variables are strictly positive.
