from mlagents_envs.environment import UnityEnvironment
import os

env_path = os.environ.get('AGENTARK_ENV_PATH')
# env_path = None connects to the Unity Editor.

# This is a non-blocking call that only loads the environment.
env = UnityEnvironment(file_name=env_path, seed=1, side_channels=[])
env.reset()
behavior_name = list(env.behavior_specs)[0]
spec = env.behavior_specs[behavior_name]

for episode in range(3):
    env.reset()
    decision_steps, terminal_steps = env.get_steps(behavior_name)
    tracked_agent = -1
    done = False
    episode_rewards = 0
    while not done:
        if tracked_agent == -1 and len(decision_steps) >= 1:
            tracked_agent = decision_steps.agent_id[0]
        # action = spec.action_spec.random_action(len(decision_steps))
        action = spec.action_spec.empty_action(1)
        env.set_actions(behavior_name, action)

        env.step()

        decision_steps, terminal_steps = env.get_steps(behavior_name)
        if tracked_agent in decision_steps:
            episode_rewards += decision_steps[tracked_agent].reward
        if tracked_agent in terminal_steps:
            episode_rewards += terminal_steps[tracked_agent].reward
            done = True
    print(f"total rewards for episode {episode} is {episode_rewards}")

env.close()
print('closed environment')
