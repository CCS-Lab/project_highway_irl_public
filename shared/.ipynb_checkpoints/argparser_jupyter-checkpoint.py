import argparse

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--savedir', help='save directory', default='trained_models_AIRL/')
    parser.add_argument('--model_save', help='save model name', default='model.ckpt')
    parser.add_argument('--reward_savedir', help="reward save directory", default='rewards_record_AIRL/')

    # expert data
    parser.add_argument('--expert_traj_dir', help="expert data directory", default='trajectory/')

    # The environment
    parser.add_argument("--envs_base", default="Highway")
    parser.add_argument("--envs_1", default="Highway_10000_take7")

    # The hyperparameter of PPO_training
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--lambda_1', default=0.95, type=float)
    parser.add_argument('--lr_policy', default=1e-4, type=float)
    parser.add_argument('--ep_policy', default=1e-9, type=float)
    parser.add_argument('--lr_value', default=1e-4, type=float)
    parser.add_argument('--ep_value', default=1e-9, type=float)
    parser.add_argument('--clip_value', default=0.1, type=float)
    parser.add_argument('--alter_value', default=False, type=bool)

    # The hyperparameter of the policy network
    parser.add_argument('--units_p', default=[64, 64, 64], type=int)
    parser.add_argument('--units_v', default=[96, 96, 96], type=int)

    # The hyperparameter of the policy training
    parser.add_argument('--iteration', default=int(10001), type=int)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--num_epoch_policy', default=6, type=int)  #6, 10
    parser.add_argument('--num_epoch_value', default=10, type=int)
    parser.add_argument('--min_length', default=5000, type=int)
    parser.add_argument('--num_parallel_sampler', default=10, type=int)
    parser.add_argument('--full_traj_length', default=1000000, type=int)
    parser.add_argument('--traj_length', default=10000, type=int)
    parser.add_argument('--random_starting', default=True, type=bool)

    # The hyperparameter of the discriminator network
    parser.add_argument('--lr_discrim', default=1e-4, type=float)

    # The hyperparameter of the discriminator training
    parser.add_argument('--num_expert_dimension', default=5000, type=int)
    parser.add_argument('--num_epoch_discrim', default=5, type=int)
    parser.add_argument('--batch_size_discrim', default=500, type=int)

    # The hyperparameter of restoring the model
    parser.add_argument('--model_restore', help='filename of model to recover', default='model.ckpt')
    parser.add_argument('--continue_s', default=False, type=bool)
    parser.add_argument('--log_file', help='file to record the continuation of the training', default='continue_C1.txt')

    return parser.parse_args([])
