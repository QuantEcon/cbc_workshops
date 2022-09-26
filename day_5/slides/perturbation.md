# Perturbation Analysis

## Quantecon workshop at Chilean Central Bank, 2022

Pablo Winant

----

## Goal

Study Neoclassical Model around the Steady-State

1. Derive First Order Conditions
2. Computing Derivatives Numerically
3. Solution Method
   - Linear Time Iteration
4. Implementation

---

## Deriving First Order Conditions

----

## Neoclassical Growth Model

<div class="container">

<div class="col">

- Transition Equation
$$\begin{eqnarray}
k_t & = & (1-\delta) k_{t-1} + i_{t-1} \\\\
z_t & = & \rho z_{t-1} + \epsilon_t
\end{eqnarray}
$$
- Definition:
$$c_t = \exp(z_t) k_t^\alpha - i_t$$

- Control $i_t\in[0, \exp(z_t)k_t^\alpha[$
  - or equivalently $c_t \in ]0, \exp(z_t) k_t^{\alpha}]$

- Shock:
  - $\epsilon_t$: i.i.d. with stdev $\epsilon_t$

</div>
<div class="col">


- Calibration:
  - $\beta = 0.96$
  - $\delta = 0.1$
  - $\gamma = 4.0$
  - $\alpha = 0.3$
  - $U(x)=\frac{x^{1-\gamma}}{1-\gamma}$

- Objective:
$$\max_{i_t} \sum_{t\geq0} \beta^t U(c_t)$$

</div>


----

### Deriving First Order Conditions

- Two methods:
    1. Bellman:
       1. Optimality Condition
       2. Enveloppe Condition
    2. Lagrangian:
- We will use the lagrangian version

----

### Lagrangian

- Initial Conditions (predetermined states): $z_0$, $k_0$
- Problem:
$$V(z_0, k_0) = \max_{\begin{matrix}i_0, i_1, i_2, \cdots \\\\c_0, c_1, c_2 \cdots \\\\ k_1, k_2, \cdots\end{matrix}} \sum_{t \geq 0}\beta^t U(c_t)$$

$$\text{s.t.}\forall t\geq 0, \\; \\; \begin{eqnarray}
\mu_t:\quad &  0 & \leq & i_t  \\\\
\nu_t:\quad &  i_t & \leq & \exp(z_t) k_t^{\alpha} \\\\
\lambda_t:\quad &  i_t & = & \exp(z_t) k_t^{\alpha} - c_t\\\\
q_t:\quad &  k_{t+1} & = & (1-\delta) k_{t} + i_{t}
\end{eqnarray}$$
- Lagrangian:
$$\mathcal{L(z_0, k_0)} =   \sum_{t \geq 0} \beta^t\left\\{ U(c_t) + \mu_t \left( i_t \right) + \nu_t \left(\exp(z_t)k_t^{\alpha} - i_t \right) + \lambda_t \left(\exp(z_t) k_t^{\alpha}  - i_t -c_t \right)  + q_t \left( (1-\delta) k_{t} + i_{t} - k_{t+1} \right) \right\\}$$

----

### Lagrangian

<div class="container">
<div class="col">

- We maximize the lagrangian to get:

$$\begin{eqnarray}
\forall t\geq0 & \frac{\partial \mathcal{L}}{\partial i_t} & = & 0 \\\\
& \frac{\partial \mathcal{L}}{\partial c_t} & = & 0 \\\\
 & \frac{\partial \mathcal{L}}{\partial k_{t+1}} & = & 0 
\end{eqnarray}$$


- It is important to note that we don't differentiate with respect to a predetermined state
  - check that you don't differentiate w.r.t. $k_0$
- It looks like the first order condition added four new variables $\mu_t$,$\nu_t$, $\lambda_t$, $q_t$

</div>

<div class="col">

- Luckily these variables are associated to slackness conditions.

|                    |                                             |
| ------------------ | ------------------------------------------- |
| $\mu_t \geq 0$     | $ i_t \geq 0$                |
| $\nu_t \geq 0$     | $\exp(z_t) k_t^{\alpha}-i_t \geq 0$         |
| $q_t \geq 0$       | $(1-\delta) k_{t} + i_{t} - k_{t+1} \geq 0$ |
| $\lambda_t \geq 0$ | $\exp(z_t) k_t^{\alpha}  - i_t -c_t = 0$    |

- The Karush-Kuhn-Tucker states, that for each slackness condition, at any time, either
  - the lagrangian is 0 and it disappears from the F.O.C.s
  - or it is not 0 and the associated constraint adds another condition

</div>
</div>

----

### Eliminating constraints

<div class="container">
<div class="col">

|                    |                                             |
| ------------------ | ------------------------------------------- |
| $\mu_t \geq 0$     | $ i_t \geq 0$                |
| $\nu_t \geq 0$     | $\exp(z_t) k_t^{\alpha}-i_t \geq 0$         |
| $q_t$       | $(1-\delta) k_{t} + i_{t} - k_{t+1} = 0$ |
| $\lambda_t$ | $\exp(z_t) k_t^{\alpha}  - i_t -c_t = 0$    |

- In general slackness conditions can be occasionally binding
- For perturbation analysis, we need constraints to be always (or never binding)

</div>
<div class="col">

Let's review them:
- $\nu_t$: it is equivalent to $c_t\geq 0$
  - we necessarily have $c_t>0$ since $U^{\prime}(0)=\infty$
  - hence $\nu_t=0$
- $\mu_t$: it states $k_{t+1}\geq 0$
  - if $k_{t+1}=0$, then $c_{t+1}$. We can conclude $k_{t+1}>0$
  - hence $\mu_t=0$
- for multipliers associated to an equality constraint, we always keep the system
  - multiplier can have any sign
- inequality formulation is sometimes found too:
  -  $c_t \leq \exp(z_t) k_t^{\alpha}  - i_t$ ( you can destroy production insead of eating or investing it)
  - $k_{t+1} \leq (1-\delta) k_{t} + i_{t}$ (you can destroy capital instead of investing it)

</div>


----

### First order model

- Optimality Condition:
$$\beta  \left[ \frac{\left(c_{t+1}\right)^{-\gamma}}{\left(c_t\right)^{-\gamma}} \left( (1-\delta + \alpha z_{t+1} k_{t+1}^{\alpha -1}) \right)\right] = 1$$
- Definition:
$$c_t = k_t^\alpha - i_t$$

- Transition:
$$k_t (1-\delta) k_{t-1} + i_{t-1}$$
$$z_t = \rho z_{t-1} + \epsilon_t$$

----

### Steady-State


<div class="container">

<div class="col">

- Steady-State: $\overline{i}, \overline{k}, \overline{z}$ such that:
  - $z_{t+1}=z_t=\overline{z}$
  - $k_{t+1}=k_t=\overline{k}$
  - $i_{t+1}=i_t=\overline{i}$
  - $c_{t+1}=c_t=\overline{i}$
- ...satisfy the first order conditions
- ...i.e.
$$\overline{c} = exp(\overline{z}){\overline{k}}^{\alpha} - \overline{i}$$
$$\beta   \left( (1-\delta + \alpha {\overline{k}}^{\alpha -1}) \right) = 1$$
$$\overline{k}=  (1-\delta) \overline{k} + \overline{i}$$
$$\overline{z} = \rho \overline{z}$$

</div>
<div class="col">

- Solution?
  - closed-form
  - numerical

- Here we can get a closed form:

$$\begin{eqnarray}
\overline{k} & = & \left( \frac{\frac{1}{\beta}-(1-\delta)}{\alpha} \right)^{\frac{1}{\alpha - 1}} \\\\
\overline{i} & = & \delta \overline{k} \\\\
\overline{c} & = & {\overline{k}}^{\alpha} - \overline{i} \\\\
\overline{z} & = & 0
\end{eqnarray}$$

</div>
</div>

----

### Perturbation Analysis

<div class="container">

<div class="col">

- Write all variables in deviation to the steady-state:
$$z_{t}=\overline{z} + \Delta z_t$$
$$k_{t}=\overline{k} + \Delta k_t$$
$$i_{t}=\overline{i} + \Delta i_t$$
$$c_{t}=\overline{c} + \Delta c_t$$
  - Remark: some smart economists use log-deviations (i.e. $x_t = \overline{x} \hat{x}_t$ to make computations easier)

</div>
<div class="col">

- Replace in the system
$$\beta  \left[ \frac{\left(\overline{c}+ \Delta c_{t+1}\right)^{-\gamma}}{\left(\overline{c} + \Delta c_t\right)^{-\gamma}} \left( (1-\delta + \alpha e^{z_{t+1}+\Delta z_{t+1}}(\overline{k} + \Delta k_{t+1})^{\alpha -1}) \right)\right] = 1$$
$$\overline{c} + \Delta c_t = (\overline{k}+ \Delta k_t)^\alpha - \overline{i} - \Delta i_t$$
$$\overline{k} + \Delta k_t = (1-\delta) (\overline{k}+ \Delta k_{t-1}) + \overline{i }+ \Delta i_{t-1}$$
$$\overline{z }+ \Delta z_t = \overline{z}+ \Delta \rho z_{t-1} + \Delta \epsilon_t$$
- Differentiate...
- (if we want to limit the number of equations, we can replace $c_t$ by its value)

</div>

----

### Result:

- Optimality Condition
$$\begin{bmatrix} . & . & . & . \\\\ . & . & . & . \end{bmatrix} \begin{bmatrix} \Delta i_t \\\\ \Delta c_t \\\\ \Delta k_t \\\\ \Delta z_t \end{bmatrix} + \begin{bmatrix} . & . & . & . \\\\ . & . & . & . \end{bmatrix}  \begin{bmatrix} \Delta i_{t+t} \\\\ \Delta c_{t+1} \\\\\Delta k_{t+1} \\\\ \Delta z_{t+1} \end{bmatrix} = \begin{bmatrix} 0 \\\\ 0 \end{bmatrix}$$

- Transition
$$ \begin{bmatrix} \Delta k_t \\\\ \Delta z_t \end{bmatrix} = \begin{bmatrix} . & .  \\\\ . & . \end{bmatrix} \begin{bmatrix} \Delta k_{t-1} \\\\ \Delta z_{t-1} \end{bmatrix}  + \begin{bmatrix} . & .  \\\\ . & . \end{bmatrix} \begin{bmatrix}\Delta i_{t-1}\\\\ \Delta c_{t-1}\end{bmatrix} + \begin{bmatrix} . \\\\ . \end{bmatrix} \begin{bmatrix} \epsilon_t \end{bmatrix}$$


----

### General formulation


- Controlled process with optimality conditions
$$\begin{eqnarray}
E_t \left[ f(s_t, x_t, s_{t+1}, x_{t+1} ) \right] & = & 0\\\\
s_t & = & g(s_{t-1}, x_{t-1},\epsilon_t )
\end{eqnarray}$$
- Variables:
  - states: $s_t \in R^{n_s}$
  - controls: $x_t \in R^{n_x}$
  - shock: $\epsilon_t \in R^{n_e}$, i.i.d. $E_{t}\left[\epsilon_{t+1}\right]=0$
- Solution:
  - decision rule: $x_t = \varphi(s_t)$


----

### Steady-state and f.o.c.

- Steady-state:
$$\begin{eqnarray}
 f(\overline{s}, \overline{x}, \overline{s}, \overline{x}) & = & 0\\\\
\overline{s} & = & g(\overline{s}, \overline{x},0)
\end{eqnarray}$$
- First order approximation (jacobians computed at s.s.):
$$\begin{eqnarray}
f^{\prime}\_{s\_{t}} \Delta s_t + f^{\prime}\_{x\_{t}} \Delta x_t + f^{\prime}\_{s\_{t+1}} \Delta s_{t+1} + f^{\prime}\_{x\_{t+1}} \Delta x_{t+1} & = & 0 \\\\
\Delta s_t & = & g^{\prime}\_{s\_{t-1}} \Delta s\_t + g^{\prime}\_{x\_{t-1}} \Delta x_t + g^{\prime}_e \Delta e_t
\end{eqnarray}$$
  - note how the expectation sign disappears (first order equivalence)
- How can we get the derivatives easily?

---

## Computing Derivatives

----

### Main approaches

- Main approaches:
    1. Manual
    2. Finite Differences
    3. Symbolic Differentiation
    4. Automatic Differentiation
- Could we use Julia to make all the hard work?
  - yes, with the right library !

----

### Manual Differentiation

- Trick:
    - never use $\frac{d}{dx} \frac{u(x)}{v(x)} = \frac{u'(x)v(x)-u(x)v'(x)}{v(x)^2}$
      - too error prone
    - use instead $$\frac{d}{dx} {u(x)v(x)} = {u'(x)v(x)+u(x)v'(x)}$$ and $$\frac{d}{dx} \frac{1}{u(x)} = -\frac{u^{\prime}}{u(x)^2}$$
- You can get easier calculations (in some cases) by using log-deviation rules

----

### Finite Differences

- Choose small $\epsilon>0$, typically $\sqrt{ \textit{machine eps}}$

- Forward Difference scheme:
    -  $f'(x) \approx \frac{f(x+\epsilon) - f(x)}{\epsilon}$
    - precision: $o(\epsilon)$
    - bonus: if $f(x+\epsilon)$ can compute $f(x)-f(x-\epsilon)$ instead (Backward)

- Central Difference scheme:
    -  $f'(x) \approx \frac{f(x+\epsilon) - f(x-\epsilon)}{2\epsilon}$
    - average of forward and backward
    - precision: $o(\epsilon^2)$

----

### Finite Differences: Higher order

- Central formula:
$$\begin{aligned}
f''(x) & \approx & \frac{f'(x)-f'(x-\epsilon)}{\epsilon} \approx \frac{(f(x+\epsilon))-f(x))-(f(x)-f(x-\epsilon))}{\epsilon^2} \\ & = & \frac{f(x+\epsilon)-2f(x)+f(x-\epsilon)}{\epsilon^2}
\end{aligned}$$
    - precision: $o(\epsilon)$
- Generalizes to higher order but becomes more and more innacurate

----

### Symbolic Differentiation

- manipulate the tree of algebraic expressions
    - implements various simplification rules
- requires mathematical expression
- can produce mathematical insights
- sometimes inaccurate:
  - cf: $\left(\frac{1+u(x)}{1+v(x)}\right)^{100}$

----

### Julia Packages:



- *FiniteDiff.jl*, *FiniteDifferences.jl*, *SparseDiffTools.jl*
    - careful implementation of finite diff
    - use it instead of your own version
- *Calculus.jl*:
    - pure julia
    - finite difference
    - symbolic calculation
- *SymEngine.jl*
    - fast symbolic calculation

----

### Automatic Differentiation

- does not provide mathematical insights but solves the other problems
  - can differentiate any piece of code
  - numerically stable
- two flavours
  - forward accumulation
  - reverse accumulation
  - ...but arbitrary mix is also possible in theory

----

### Simple example:

```julia
function f(x::Number)
    a = x + 1
    b = x^2
    c = sin(a) + b
    c
end
```

----

### Automatic rewrite: source code transform

```julia
function f(x::Float64)

    # x is an argument
    x_dx = 1.0

    a = x + 1
    a_dx = x_dx

    b = x^2
    b_dx = 2*x*x_dx

    t = sin(a)
    t_dx = cos(a)*a_dx

    c = t + b
    c_x = t_dx + b_dx

    return (c, c_x)
end
```

----

### Dual numbers: operator overloading

```julia
struct DN<:Number
    x::Float64
    dx::Float64
end
import Base: +, -, *, /, sin

+(a::DN,b::DN) = DN(a.x+b.x, a.dx+b.dx)
-(a::DN,b::DN) = DN(a.x-b.x, a.dx-b.dx)
*(a::DN,b::DN) = DN(a.x*b.x, a.x*b.dx+a.dx*b.x)
/(a::DN,b::DN) = DN(a.x/b.x, (a.dx*b.x-a.x*b.dx)/b.dx^2)
sin(a::DN) = DN(sin(a.x), cos(a.x)*a.dx)
# finish the work
```

----

### Compatible with control flow

```julia
import ForwardDiff: Dual

x = Dual(0.5, 1.0)
b = sum([(x)^i/i*(-1)^(i+1) for i=1:500])
# compare with log(1+x)
```
  - generalizes nicely to gradient computations

```julia
x = Dual(1.0, 1.0, 0.0)
y = Dual(1.0, 0.0, 1.0)
exp(x) + log(y)
```

----

### Forward Accumulation Mode

- Forward Accumulation mode:-
  - isomorphic to dual number calculation
  - and to manual application of chain rule
  - compute tree with values and derivatives at the same time
  - efficient for $f: R^n\rightarrow R^m$, with $n<<m$
    - (keeps lots of empty gradients when $n>>m$)

----

### Reverse Accumulation Mode

- Reverse Accumulation / Back Propagation
    - efficient for $f: R^n\rightarrow R^m$, with $m<<n$
    - requires data storage (to keep intermediate values)
    - graph / example

- Very good for machine learning:
   - $\nabla_{\theta} \Xi(\theta)$ where $\Xi$ is a loss/objective function $\in R$

----

### Libraries for AutoDiff

- See JuliaDiff: http://www.juliadiff.org/
  - *ForwardDiff.jl*
  - *ReverseDiff.jl*
- *Zygote.jl*
- Future™: *Enzime.jl*
- Deep learning framework:
  - *Flux.jl*
  - higher order diff w.r.t. any vector -> tensor operations:
    - *Tensorflow.jl*

---

## Solution using Linear Time iteration

----

### Our problem

<div class="container">

<div class="col">

- General formulation of a *linearized* model:
$$ \begin{eqnarray} A s_t + B x_t + C s_{t+1} + D x_{t+1} & = & 0_{n_x} \\\\
s_{t+1} & = & E s_t + F x_t  + G e_{t+1}\end{eqnarray}$$
where:
  - $s_t \in \mathbb{R}^{n_s}$ is a vector of *states*
  - $x_t \in \mathbb{R}^{n_x}$ is a vector of *controls*
- Remark:
  - first equation is *forward* looking
  - second equation is *backward* looking

</div>

<div class="col">
<div>

- In the neoclassical model:
$$\begin{eqnarray}
s_t & = & (\Delta z_t, \Delta k_t) \\\\
x_t & = & (\Delta i_t, \Delta c_t)
\end{eqnarray}$$

- The linearized system is:
$$\begin{eqnarray}
A & = & ...\\\\
B & = & ...\\\\
C & = & ...\\\\
D & = & ...\\\\
E & = & ...\\\\
F & = & 
\end{eqnarray}$$

</div>
</div>

----

### Solution

<div class="container">
<div class="col">

- What is the solution of our problem?
- At date $t$ *controls* must be chosen as a function of (predetermined) *states*

- Mathematically speaking, the solution is a function $\varphi$ such that:
  $$\forall t, x_t = \varphi(s_t)$$
- Since the model is linear we look for un unknown matrix $X \in \mathbb{R}^{n_x} \times \mathbb{R}^{n_s}$ such that:

$$\Delta x_t = X \Delta s_t$$


</div>

<div class="col">
<div>
  
  In the neoclassical model
- The control $i_t$ (resp $c_t$) is a function of states $k_t, z_t$
$$\begin{eqnarray}
i_t = i(z_t, k_t)\\\\
c_t = c(z_t, k_t)
\end{eqnarray}$$
- In the linearized model:
$$\begin{eqnarray}\Delta i_t =i_z \Delta z_t + i_k \Delta k_t\\\\
\Delta c_t =c_z \Delta z_t + c_k \Delta k_t\end{eqnarray}$$
- Or
$$\begin{bmatrix}\Delta i_t \\\\ \Delta c_t  \end{bmatrix} = \underbrace{\begin{bmatrix}\ i_z & i_k  \\\\ \Delta c_z & c_k  \end{bmatrix} }_{X}\begin{bmatrix}\Delta z_t \\\\ \Delta z_k  \end{bmatrix} $$


</div>
</div>


----

### Optimality condition:


<div class="container">

<div class="col">

- Replacing in the system (assuming $\Delta e_t=0$)
$$ \begin{eqnarray}
\Delta x_t & = & X \Delta s_t \\\\
\Delta s_{t+1} & = & E \Delta s_t + F X \Delta s_t \\\\
\Delta x_{t+1} & = & X \Delta s_{t+1} \\\\
A \Delta s_t + B \Delta x_t + C \Delta s_{t+1} + D \Delta x_{t+1} & = & 0
\end{eqnarray}
$$

- If we make the full substitution:

$$( (A + B X) + ( D X + C) ( E  + F X ) ) s_t = 0$$

</div>
<div class="col">

- This must be true for all $s_t$. We get the special Ricatti equation:

$$(A + B \color{red}{X}) + ( D \color{red}{X} + C) ( E  + F \color{red}{X} ) = 0 $$

- this is a __quadratic__, __matrix__  equation( $X$ is 2 by 2 ):
  - requires special solution method
  - there are multiple solutions: which should we choose?
    - today: linear time iteration selects only one solution
    - next time: eigenvalues analysis

</div>
</div>

----

### Linear Time Iteration

- Let's be more subtle: define
  - $X$: decision rule today and
  - $\tilde{X}$ is decision rule tomorrow.
$$\begin{eqnarray}
\Delta x_t & =&  X \Delta s_t \\\\
\Delta s_{t+1} & = & E \Delta  s_t + F X \Delta s_t \\\\
\Delta x_{t+1} & = & \tilde{X} \Delta s_{t+1} \\\\
A \Delta s_t + B \Delta x_t + C \Delta s_{t+1} + D \Delta x_{t+1} & = & 0
\end{eqnarray}$$
- We get, $\forall s_t$:
$$(A + B X) + (C + D \tilde{X}) ( E  + F X ) ) \Delta s_t = 0 $$
- Again, this must be zero in all states $\Delta s_t$.

----

### Linear Time Iteration (2)


<div class="container">

<div class="col">

- We get the equation: 
$$\begin{eqnarray}F(X, \tilde{X}) & = & (A + B X) + ( C+ D \tilde{X}) ( E  + F X ) \\\\ & = &  0\end{eqnarray}$$
- Consider the *linear time iteration* algorithm
- When the model is well-specified it is guaranteed to converge to the right solution.
  - cf *linear time iteration* by Pontus Rendahl ([link](https://irihs.ihs.ac.at/id/eprint/4351/1/es-330.pdf))
- There are simple criteria to check that the solution is right, and that the model is well specified
- $T$ is the time iteration operator... for linear models
  - it does *forward iteration* ($X_t$ as a function of $X_{t+1}$)

</div>
<div class="col">

- algorithm:
  - choose stopping criteria: $\epsilon_0$ and $\eta_0$
  - choose random $X_0$
  - given $X_n$:
    - compute $X_{n+1}$ such that $F(X_{n+1}, X_{n}) = 0$
      <span class="r-stack"><span class="fragment current-visible">$$(B + (C+D X_{n})F)X_{n+1} + A  + (C+D X_n )E=0$$</span><span class="fragment current-visible">$$X_{n+1} = (B + (C + D X_n) F)^{-1} (A + (C+DX_n)E)$$</span><span class="fragment">$$X_{n+1} = T(X_n)$$</span></span>
    - compute:
      - $\eta_n = |X_{n+1} - X_n|$
      - $\epsilon_n = F(X_{n+1}, X_{n+1})$
    - if $\eta_n<\eta_0$ and $\epsilon_n<\epsilon_0$
      - stop and return $X_{n+1}$
      - otherwise iterate with $X_{n+1}$

</div>
</div>

----

### Simulating the model

- Suppose we have found the solution $\Delta x_t  = X \Delta s_t$
- Recall the transition equation: $\Delta s_{t+1} = F \Delta s_t + G \Delta x_t + Q \Delta e_t$
- We can now compute the model evolution following initial deviation in the state:
$$\Delta s_t = \underbrace{(F + G X)}\_{P} \Delta s\_{t-1} + Q \Delta e_t $$
  - $P$ is the simulation operator
  - it is a *backward* operator
(TODO: example of a reaction to a shock)
- The system is stable if the biggest eigenvalue of $P$ is smaller than one...
- ... or if its spectral radius is smaller than 1:
  $$\rho(P)<1$$
- This condition is called *backward* stability
  - it rules out *explosive solutions*
  - if $\rho(P)>1$ one can always find $s_0$ such that the model simulation diverges

----

### Spectral radius

<div class="container">
<div class="col">

- How do you compute the spectral radius of matrix P?
  - naive approach: compute all eigenvalues, check the value of the biggest one...
  - better approach: power iteration method
- Power iteration method:
  - take a linear operator $L$ over a Banach Space $\mathcal{B}$ (vector space with a norm)
  - use the fact that *for most* $u_0\in \mathcal{B}$, $\frac{|L^{n+1} u_0|}{|L^n u_0|}\rightarrow \rho(L)$

</div>
<div class="col">

- Algorithm:
  - choose tolerance criterium: $\eta>0$
  - choose random initial $x_0$ and define $u_0 = \frac{x_0}{|x_0|}$
    - by construction: $|u_0|=1$
  - given $u_n$, compute 
    - $x_{n+1} = L.u_n$
    - $u_{n+1} = \frac{x_{n+1}}{|x_{n+1}|}$
    - compute $\eta_{n+1} = |u_{n+1} - u_n|$
    - if $\eta_{n+1}<\eta$: 
      - stop and return $|x_{n+1}|$
      - else iterate with $u_{n+1}$

</div>
</div>

----

### Stability of the backward operator


- To solve the model we use the backard operator: $$T: \begin{eqnarray} \mathbb{R}^{n_x} \times \mathbb{R}^{n_s}  & \rightarrow &  \mathbb{R}^{n_x} \times \mathbb{R}^{n_s}  \\\\X_{t+1} & \mapsto & X_t =0\end{eqnarray} \text{so that} F(X_t,X_{t+1}) $$
- What about its stability?
- Recall: fixed point $\overline{z}$ of recursive sequence $z_n=f(z_{n_1})$ is stable if $|f^{\prime}(\overline{z})|<1$
- We need to study $T^{\prime}$ of ($X$).
  - but $T$ maps a matrix to another matrix 😓
  - how do we differentiate it? 🐉
  
<img class="fragment" src="adele_bs.jpg">

----

### Differentials

- Consider a Banach Space $\mathcal{B}$.
- Consider an operator (i.e. a function): $\mathcal{T}: \mathcal{B} \rightarrow \mathcal{B}$.
- Consider $\overline{x} \in \mathcal{B}$.
- $\mathcal{T}$ is differentiable at $\overline{x}$ if there exists a bounded linear operator $L \in \mathcal{L}(\mathcal{B})$ such that:
$$\mathcal{T}(x) = \overline{x} + L.(x-\overline{x}) + o(|x-\overline{x}|)$$
  - when it exists we denote this operator by $\mathcal{T}^{\prime}(\overline{x})$
- Remarks:
  - Bounded operator means: $\sup_{|x|=1} |L.x|<+\infty$
  - This definition of a derivative is usually referred to as Fréchet-derivative
  - In finite-dimensional spaces, all linear operator are bounded

----

### Differentials of linear operators

- $x$ vector, A a matrix, $T(x) = Ax$
  - then $T(x+u) = Ax + A.u + 0$
  - $T^{\prime}(x).u = A u$ 
  - $T^{\prime}(x) = A$ for all x
- $A$ a matrix, $X$ a matrix: $T(X) = A X$
  - then $T(X+u) = A X  + A u  + 0$
  - $T^{\prime}(X).u = A u$
- $A$ a matrix, X a matrix: $T(X) = A X B$
  - then $T(X+u) = A X B + A u B$
  - $T^{\prime}(X).u = A u B$

----

### Back to the time iteration operator

- $T(X)$ is implicitly defined by $F(T(X), X)=0$
- $F(X,Y) = (A + B X) + ( C+ D Y) ( E  + F X )$
  - it is linear in $X$ and in $Y$
- $F^{\prime}_X (X, Y).u = (B + (C+DY)F) u$
  - a regular matrix multiplication
  - its inverse is: $F^{\prime}_X (X, Y)^{-1} = (B + (C+DY)F)^{-1}$
- $F^{\prime}_Y (X, Y).u = D u (E+FX)$
  - a linear operation on matrices

----

### The derivative of the time-iteration operator

- Implicit relation $F(T(X), X)$ can be differentiated:
$$F^{\prime}_X (T(X), X) . T^{\prime} (X) + F_Y^{\prime}(T(X),X) = 0$$
- $F^{\prime}_X (T(X), X)$ being a regular matrix, it is (conceptually) easy to invert:
$$T^{\prime}(X) = -(F^{\prime}_X (T(X), X))^{-1}F_Y^{\prime}(T(X),X)$$
- Finally, we get the explicit formula for the linear operator $T^{\prime}$ computed at the steady state:
$$T^{\prime}(\overline{X}).u = ((B + (C+D \overline{X})F)^{-1})D u (E+F \overline{X})$$
- We can compute the spectral radius of $T^{\prime}(\overline{X})$ using the power iteration method

----

### Recap

1.  We compute the derivatives of the model
2.  Time iteration algorithm, starting from an initial guess $X_0$ and we repeat until convergence:
$$X_{n+1} = (B + (C + D X_n) F)^{-1} (A + (C+DX_n)E)$$
3.  Check model is well defined using the power method:
    -  backward stability: derivative of simulation operator
      $$\rho(\underbrace{E + F \overline{X}}_{P} ) < 1$$
    -  forward stability: derivative of time iteration operator
      $$\rho \left( u\mapsto ((B + (C+D \overline{X})F)^{-1})D u (E+F \overline{X}) \right) < 1$$
    -  These two are equivalent to the so-called Blanchard-Kahn conditions.
4. Simulate using:
  $$\begin{eqnarray} \Delta s_t & = & P \Delta s_{t-1} + Q e_t \\\\ x_t & = & X \Delta s_t\end{eqnarray}$$
