# highway_irl

"highway_task" folder contains the task. Follow the instructions in the readme file in the folder to install and run the task.

<h2>Analysis code description:</h2>

<h3>Python files</h3>

1_install_env: install required packages (requires setuptools==62.6.0 & wheel==0.38.0).

2_dqn_train_exp_iter: trains a DQN model

3_dqn_collect_batch: generate behaviors using a DQN model

4_run_highway_AIRL: AIRL for the data from DQN

5_data_to_traj: generate state-action trajectories from the task data

6_run_AIRL_data: AIRL for the data from experiments

<h3>IPython files</h3>

descriptive_analysis_final: shows the correlation between BIS and behavioral task measures

IRL_group: shows the model fit of the AIRL models

IRL_raw_reward: illustrates the IRL rewards in simplified state spaces

reward_trajectory: illustrates the IRL reward trajectories for overtaking and crashing
