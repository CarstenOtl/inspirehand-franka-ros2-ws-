# FR3 MuJoCo model

`fr3.xml` is the `franka_fr3` model from
[mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie) (Apache-2.0, see
`LICENSE.menagerie`), itself derived from `franka_description`. Changes made here:

- the 55 MB of visual OBJ meshes are not shipped; the STL collision meshes are drawn instead.
  Drop the Menagerie `assets/*.obj` files in and restore the `visual` geoms if you want the
  pretty version;
- the position actuators are replaced by torque `<motor>`s, so that the control law is always
  the one in `franka_mujoco_hardware` or in the offline simulator, never MuJoCo's own PD;
- 1 ms timestep to match the 1 kHz control cycle;
- a `flange` site at the `fr3_link8` frame of the URDF (0.107 m along z of link 7);
- the `home` keyframe is the FR3 ready pose `[0, -pi/4, 0, -3pi/4, 0, pi/2, pi/4]`.
