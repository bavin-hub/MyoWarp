# H1–OSL model

This model keeps the Unitree H1 body, left leg, and right hip yaw/roll/pitch chain, then replaces the H1 right knee and ankle chain with the OSL prosthesis.

- `h1_osl.xml`: combined robot model.
- `scene.xml`: viewable scene with a floor and a compatible home keyframe.
- `assets/right_hip_pitch_link_mod.stl`: modified right H1 thigh supplied for the integration.
- `assets/osl_*.stl`: copied OSL geometry.

At the zero pose, the OSL knee axis is positioned at the original H1 right-knee axis `(0, 0, -0.4)` in the right hip-pitch frame. The OSL knee axis and the H1 knee axis therefore coincide, and the two feet have approximately the same ground height.
