# easy-gravity

A naive way to extract gravity from poses

See [here](https://twitter.com/Parskatt/status/1815262368345256319) for some alternative methods


## Method
Assume cameras are not in plane rotated. This means that gravity must lie on the "pointing/up" plane for each camera.
For any vector $x$ we can easily compute the squared distance to this plane as $x \cdot \hat{n}$. The normals of the plane is simply the x-direction of the camera. Differentiate this sum and solve for 0 you get an eigenvalue eq. Solve for smallest eigenvalue.