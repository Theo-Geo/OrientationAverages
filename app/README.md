# Average for Crystal Orientation

# Theoretical basis

Any rotation in 3D can be represented by a 3x3 orthogonal matrix that preserves orientation, formally the group $SO(3)$. For a crystal with certain symmetries there are operations that leave the crystal orientation invariant. For example, a 180° rotation around $z$ is defined by the matrix $\mathbf{P}_z = \mathrm{diag}(-1,-1,1)$, and would leave a crystal with monoclinic symmetry around $z$ invariant. Following ([Man, 2022](https://doi.org/10.1007/s10659-022-09882-8) Chapter 6), one can define the set of all symmetry operations as,
```math
G_{cr} = \{\mathbf{P}_1,\ \dots ,\ \mathbf{P}_{N_{cr}}\} \subset SO(3) \text{ ,}
```
with $\mathbf{P_1}=\mathbf{I}$ and $N_{cr}$ the number of crystal symmetries. Each crystal orientation then has $N_{cr}$ representations such that one crystal orientation is defined by the set,
```math
\mathbf{R}G_{cr} = \{ \mathbf{R}\mathbf{P}_i:P_i\in G_{cr} \} \text{ .}
```
Formally, the space of crystal orientation is then defined as a quotient set of the space of rotations,
```math
SO(3)/G_{cr} = \{\mathbf{R}G_{cr}:\mathbf{R}\in SO(3)\} \text{ .}
```
We can choose one specific representation of the quotient space, e.g. in terms of a range of euler angles. Then we can send all rotation into this range of values by applying the symmetry operations. In crystallography people say they send all rotations into a fundamental zone. A fundamental zone is not unique and can be centered around any point in the space of rotations.

To define an average between crystal orientations we need to define a distance between two objects in the space $SO(3)/G_{cr}$. If we have two orientations $\mathbf{R}_1G_{cr}$ and $\mathbf{R}_2G_{cr}$, we find a distance by picking the symmetry operations $\mathbf{P}_i$ of one of the crystal orientations such that it is closest to the other. We can also say we send one crystal orientation into the fundamental zone around the other orientation. Following ([Man, 2022](https://doi.org/10.1007/s10659-022-09882-8) Chapter 6.4) we can write,
```math
d_{SO(3)/G_{cr}} (\mathbf{R}_1G_{cr}, \mathbf{R}_2G_{cr}) = \min_{\mathbf{P} \in G_{cr}} d(\mathbf{R}_1, \mathbf{R}_2\mathbf{P} ) \text{ ,}
```
here $d(\cdot, \cdot)$ is a metric in the space of rotations.

## Definition of an Average

An average in the space of rotations is defined as the rotation, which minimizes the squared distance to all samples (e.g. [moakher, 2002](https://doi.org/10.1137/S0895479801383877)),
```math
:label: eqn:rotation_average
\bar{\mathbf{R}} =  \underset{\mathbf{R}\in SO(3)}{\rm argmin}\sum_n d^2(\mathbf{R}_i,\mathbf{R}) \text{ .}
```
The two definitions of the metric distance we consider are the euclidean distance (e.g. straight line through the ambient space),
```math
:label: eqn:rotation_euclidean_norm
d_{\rm{eucl.}}(\mathbf{R}_1,\mathbf{R}_2) = ||\mathbf{R}_1- \mathbf{R}_2||_F
```
and the geodesic or Riemannian distance (e.g. arc following the curvature of the space),
```math
:label: eqn:rotation_riemann_norm
d_{\rm{geod.}}(\mathbf{R}_1,\mathbf{R}_2) =  ||\log(\mathbf{R}_1^T\mathbf{R}_2)||_F \text{ .}
```
In both cases $|| \cdot ||_F$ denotes the Frobenius norm, $||\mathbf{A}||_F:= \sqrt{\mathrm{Tr}(\mathbf{A}^T \mathbf{A}) }$. For a general crystal symmetry, the minimization problem we have to solve becomes,
```math
:label: eqn:rotation_symmetry_average
\bar{\mathbf{R}} =  \underset{\mathbf{R}\in SO(3)}{\rm argmin}\sum_i d_{SO(3)/G_{cr}}^2(\mathbf{R}_i G_{cr},\mathbf{R}G_{cr}) = \underset{\mathbf{R}\in SO(3)}{\rm argmin}\sum_i \min_{\mathbf{P}_i \in G_{cr}} d^2(\mathbf{R}_i\mathbf{P}_i,\mathbf{R}) \text{ .}
```
This is a combinatoric optimization problem for finding the optimal fundamental zone where all rotations are closest to one another. We can also rewrite the problem as a minimization problem for the objective function,
```math
:label: eqn:objective_function
F(\mathbf{R}, \{\mathbf{P}_i\}) =  \sum_i d^2(\mathbf{R}_i\mathbf{P}_i,\mathbf{R}) \text{ .}
```
With the objective function we can compare different estimates for the average.

## Quaternions

To efficiently solve this minimization problem we represent rotations by unit quaternions, $\mathbf{q} \in \mathbb{H}$. Similar to complex numbers, quaternions can be seen as 4-dimensional numbers, $\mathbf{q} = w + x\mathbf{i} + y\mathbf{j} + z\mathbf{k}$. The quaternion algebra (e.g. $\mathbf{i}\mathbf{j} = \mathbf{k}$) defines the quaternion product $\circ$, which can be related to an active rotation of a vector $\mathbf{v}$. By representing the vector as a quaternion with no scalar part $\mathbf{v} = 0 + v_x\mathbf{i} + v_y\mathbf{j} + v_z\mathbf{k}$, we can write,
```math
\mathbf{v}' = \mathbf{R}\cdot \mathbf{v} = \mathbf{q}\circ \mathbf{v} \circ \mathbf{q}^{-1} \text{ .}
```
With some algebra we can write the right-hand side as a rotation matrix and find the function $\mathbf{R}(\mathbf{q})$ and the backwards transform $\mathbf{q}(\mathbf{R})$ (https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation). Physically, unit quaternions are an axis-angle representation of a rotation. The scalar part gives $\cos(\alpha/2)$ and the vector part gives $\sin(\alpha/2)\mathbf{n}$. Here $\alpha$ is the rotation angle and $\mathbf{n}$ is the normalized rotation axis. Each rotation can then be represented by a quaternion, $\mathbf{q}$ or $-\mathbf{q}$, because rotating by the negative angle around the negative axis is exactly the same. Therefore, quaternions are a double cover of the space of rotations, and we will work in the half-space where $q.w > 0$.
Just as for rotation matrices we can write a rotation by first $\mathbf{q}_1$ and then $\mathbf{q}_2$ as $\mathbf{q}' = \mathbf{q}_2 \circ \mathbf{q}_1$. If we rewrite the symmetry operations in terms of quaternions, $\mathbf{g}_i = \mathbf{q}(\mathbf{P}_i)$, one crystal orientation then becomes the set $\mathbf{q}G_{cr} = \{\mathbf{q}\circ\mathbf{g}_i:g_i \in G_{cr} \}$. For orthotropic symmetry the set of symmetry operations is (e.g. [Kagan, 1991](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1365-246X.1991.tb06343.x)), $G_{\rm ortho.} = \{\pm 1, \pm \mathbf{i}, \pm \mathbf{j}, \pm \mathbf{k}\}$.

Quaternions also enable us to efficiently compute the Riemannian distance as ([Huynh, 2009](https://doi.org/10.1007/s10851-009-0161-2)),
```math
d_{\rm geod.}(\mathbf{q}_1, \mathbf{q_2}) = 2\arccos(|\mathbf{q}_1\cdot\mathbf{q}_2|) \text{ .}
```
With the riemannian metric and the symmetry operations we can define the objective function in terms of quaternions as,
```math
F(\bar{\mathbf{q}}, \{\mathbf{g}_i\}) = \frac{1}{\sum_j w_j}\sum_i w_i[\ 2\arccos(|\bar{\mathbf{q}}\cdot(\mathbf{q}_i\circ\mathbf{g}_i)|)\ ]^2 \text{ ,}
```
where $\cdot$ is the dot product for vectors in $\mathbb{R}^4$. This is a slight generalization, as $w_i$ are weights associated with each rotation (quaternion). In the future, a distance weighting could be implemented based on these.

Quaternions are also convenient as there exists a well defined average. [Markley et. al. 2007](https://arc.aiaa.org/doi/10.2514/1.28949) show that the minimization Problem defined in Equ. {math:numref}`eqn:rotation_average` for the euclidean norm Equ. {math:numref}`eqn:rotation_euclidean_norm` can be reformulated into,
```math
    :label: eqn:markley_average_maximization
    \bar{\mathbf{q}} = \underset{\mathbf{q}\in \mathbb{H}}{\rm argmax} \ \mathbf{q}^T\mathbf{K}\mathbf{q}
```
with the 4x4 traceless matrix,
```math
    \mathbf{K} = 4\sum_i w_i\mathbf{q}_i\mathbf{q}_i^T - \mathbf{I}_4\sum_i w_i \text{ .}
```
The eigenvector corresponding to the maximum eigenvalue of $\mathbf{K}$ solves the maximization problem Equ. {math:numref}`eqn:markley_average_maximization`. As the matrix is traceless the largest eigenvalue will always be positive. I will give the result of this procedure the name $\bar{\mathbf{q}}_{\rm mk07}$ and define it as,
```math
    \bar{\mathbf{q}}_{\rm mk07}(\{\mathbf{q}_i\}, \{w_i\}) =  \underset{\mathbf{q}\in \mathbb{H}}{\rm argmin} \sum_i w_i d^2_{\rm eucl.}(\mathbf{R}(\mathbf{q}_i), \mathbf{R}(\mathbf{q}))
```
**Note: This average is valid for general rotations** or crystals with triclinic symmetry (no symmetry objects except the identity). In the interpolator this function is called markley average.

## Algorithm

I propose the following algorithm to solve the optimization problem {math:numref}`eqn:rotation_symmetry_average`. Our mean value and the set of symmetry operations will be denoted as $\bar{\mathbf{q}}^n$ and $\{\mathbf{g}^n_i\}$ for each iteration $n$. $n$ is restricted to a maximum number of iterations set to 50 as I have not seen more than 20 iteration for 1000 crystal orientations.

  1. Pick an initial guess $\bar{\mathbf{q}}^0$ and compute the objective function $F^0(\bar{\mathbf{q}}^{0}, \{\mathbf{g}_i^0\})$ with $\mathbf{g}_i^0 = (1, \mathbf{0})$ (no symmetrization).
  2. Project all rotations into the fundamental zone around the guess.
     - For each quaternion $\mathbf{q}_i$, compute the geodesic distances $d_{\rm geod.}(\mathbf{q}_iG_{cr}, \bar{\mathbf{q}}^n)$ for all elements in the set $\mathbf{q}_i G_{cr}$.
     - Pick $\mathbf{q}_i' = \mathbf{q}_i\circ\mathbf{g}_i^n$ such that $d_{\rm geod.}(\mathbf{q}_i', \bar{\mathbf{q}}^n)$ is minimal. In other words, find the representation in $\mathbf{q}_i G_{cr}$ that is closest to the current mean value.
  3. Compute the markley average $\bar{q}^{n+1} = \bar{\mathbf{q}}_{\rm mk07}(\{\mathbf{q}_i'\})$.
  4. Compute the objective function $F^{n+1}(\bar{\mathbf{q}}^{n+1}, \{\mathbf{g}_i^n\})$.
  5. Check if the objective function got smaller, $F^n > F^{n+1}$ ?
     - true : repeat 2,3 and 4
     - false: return $\bar{\mathbf{q}}^{n+1}$