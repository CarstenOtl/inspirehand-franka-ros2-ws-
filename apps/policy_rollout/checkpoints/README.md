# Checkpoint placeholder

Place the future, replay-qualified FR3 DP3 offline-flow student checkpoint in
this directory. Checkpoints are intentionally ignored because none of the
retained `forgeUltra/franka-chi` results qualifies as a physical FR3 policy:
the DP3 sequential student completed three of six simulated cycles before
losing its grip, and the later RealSense run pretrained only the vision head.

The app accepts a ForgeUltra checkpoint dictionary containing `config`,
`model`, and optionally `ema_model`; EMA is selected by default. It rejects
anything except the 9-D OSC, 29-D proprio, RGB point-cloud DP3 contract.
