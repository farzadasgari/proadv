# Kinetic Turbulent Energy Function

This function compute the `kinetic turbulent energy` based on velocity components.

There are three parameters in this function:

- **u (array_like)**: Array containing longitudinal velocity component. 
- **v (array_like)**: Array containing transverse velocity component. 
- **w (array_like)**: Array containing vertical velocity component. 


# Reynolds Stresses Function

This function compute the Reynolds stresses based on velocity components.
Also, it returns Tuple containing the Reynolds stresses (uu, vv, ww, uv, uw, vw).

There are three parameters in this function:

- **u (array_like)**: Array containing longitudinal velocity component. 
- **v (array_like)**: Array containing transverse velocity component. 
- **w (array_like)**: Array containing vertical velocity component. 