# SpikeCEIL
## Introduction
This is a spiking neural network implementation of Generalized Contextual Imitation Learning([CEIL](https://arxiv.org/abs/2306.14534)). We made minimum modifcations to original CEIL training code. Besides, we
provide such baseline algorithms that were compared with CEIL on environments and different combinations (lfd/lfo online/offline), which is absent in original code.

## Instruction
### Preparation
1. Training expert policy `1run_learn_expert_policy.py`
2. Cellect demotrations `2run_gen_expert_data.py`
3. Training CEIL/SpikeCEIL/Baseling by Imitation Learning `3train_ceil.py` `3train_ceil_debug_spike.py` `3train_ceil_debug_baseline.py`

### Examples
See `scripts` for mujoco environment training
