import random
import time
from dataclasses import dataclass

import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

# Import your custom environment
from gym_env_new import ClusterOptimizationEnv

@dataclass
class Args:
    exp_name: str = "dqn_custom_env"
    """The name of this experiment"""
    seed: int = 1
    """Seed of the experiment"""
    torch_deterministic: bool = True
    """If toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """If toggled, cuda will be enabled by default"""
    track: bool = False
    """If toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """The wandb's project name"""
    wandb_entity: str = None
    """The entity (team) of wandb's project"""
    capture_video: bool = False
    """Whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """Whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """Whether to upload the saved model to Hugging Face"""
    hf_entity: str = ""
    """The user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    total_timesteps: int = 500000
    """Total timesteps of the experiment"""
    learning_rate: float = 2.5e-4
    """The learning rate of the optimizer"""
    buffer_size: int = 10000
    """The replay memory buffer size"""
    gamma: float = 0.99
    """The discount factor gamma"""
    tau: float = 1.0
    """The target network update rate"""
    target_network_frequency: int = 500
    """The timesteps it takes to update the target network"""
    batch_size: int = 128
    """The batch size of samples from the replay memory"""
    start_e: float = 1.0
    """The starting epsilon for exploration"""
    end_e: float = 0.05
    """The ending epsilon for exploration"""
    exploration_fraction: float = 0.5
    """The fraction of `total_timesteps` it takes from start_e to end_e"""
    learning_starts: int = 10000
    """Timestep to start learning"""
    train_frequency: int = 10
    """The frequency of training"""

# Define the Q-Network
class QNetwork(nn.Module):
    def __init__(self, observation_space_shape, action_space_n):
        super(QNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(observation_space_shape, 256),
            nn.ReLU(),
            nn.Linear(256, action_space_n)
        )

    def forward(self, x):
        return self.network(x)

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    """Linear schedule for epsilon decay"""
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

if __name__ == "__main__":
    args = tyro.cli(Args)
    run_name = f"dqn_{args.exp_name}_{args.seed}_{int(time.time())}"
    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % (
            "\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])
        ),
    )

    # Set random seeds and device
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Instantiate the environment
    env = ClusterOptimizationEnv()
    action_space = env.action_space
    observation_space = env.observation_space

    # Remove env.seed(args.seed), handling seeding in reset
    assert isinstance(action_space, gym.spaces.Discrete), "Only discrete action spaces are supported"

    # Initialize the networks
    q_network = QNetwork(observation_space.shape[0], action_space.n).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork(observation_space.shape[0], action_space.n).to(device)
    target_network.load_state_dict(q_network.state_dict())

    # Initialize the replay buffer
    rb = ReplayBuffer(
        args.buffer_size,
        env.observation_space,
        env.action_space,
        device,
        handle_timeout_termination=False
    )

    # Start the training
    obs, _ = env.reset(seed=args.seed)
    start_time = time.time()
    for global_step in range(args.total_timesteps):
        # Epsilon-greedy action selection
        epsilon = linear_schedule(
            args.start_e, args.end_e, 
            args.exploration_fraction * args.total_timesteps,
            global_step
        )
        if random.random() < epsilon:
            action = action_space.sample()
        else:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = q_network(obs_tensor)
            action = int(torch.argmax(q_values, dim=1).cpu().numpy()[0])

        # Step the environment
        next_obs, reward, done, truncated, info = env.step(action)

        # Add to replay buffer
        rb.add(obs, next_obs, action, reward, done, info)

        # Handle terminal state
        if done or truncated:
            obs, _ = env.reset()
        else:
            obs = next_obs

        # Start training after certain number of steps
        if global_step > args.learning_starts and global_step % args.train_frequency == 0:
            # Sample from replay buffer
            data = rb.sample(args.batch_size)

            # Compute targets
            with torch.no_grad():
                next_q_values = target_network(data.next_observations).max(dim=1)[0]
                td_target = data.rewards.flatten() + args.gamma * next_q_values * (1 - data.dones.flatten())

            # Compute loss
            q_values = q_network(data.observations).gather(1, data.actions).squeeze()
            loss = F.mse_loss(q_values, td_target)

            # Optimize the model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Logging
            if global_step % 1000 == 0:
                writer.add_scalar("loss/td_loss", loss.item(), global_step)
                writer.add_scalar("loss/q_values", q_values.mean().item(), global_step)
                print(f"Step: {global_step}, Loss: {loss.item()}")

        # Update target network
        if global_step % args.target_network_frequency == 0:
            target_network.load_state_dict(q_network.state_dict())

        # Additional logging
        if global_step % 10000 == 0:
            elapsed_time = time.time() - start_time
            sps = int(global_step / elapsed_time)
            writer.add_scalar("charts/SPS", sps, global_step)
            print(f"Global Step: {global_step}, SPS: {sps}")

    # Save the model if required
    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.pth"
        torch.save(q_network.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    env.close()
    writer.close()
