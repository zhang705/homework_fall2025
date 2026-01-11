1) See hw1 if you'd like to see installation instructions. You do NOT have to redo them.
2) See the PDF for the rest of the instructions.

Run DQN with three different seeds on LunarLander-v2:

```bash
python cs285/scripts/run_hw3_dqn.py -cfg experiments/dqn/lunarlander.yaml --seed 1
python cs285/scripts/run_hw3_dqn.py -cfg experiments/dqn/lunarlander.yaml --seed 2
python cs285/scripts/run_hw3_dqn.py -cfg experiments/dqn/lunarlander.yaml --seed 3
python cs285/scripts/run_hw3_dqn.py -cfg experiments/dqn/lunarlander.yaml --seed 4
```

保存视频
```bash
python cs285/scripts/run_hw3_dqn.py -cfg experiments/dqn/lunarlander.yaml --seed 1 --num_render_trajectories 2
```


