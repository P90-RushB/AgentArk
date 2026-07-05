'''原始 mlagents-env,用于熟悉mlagents-env的api
'''
from mlagents_envs.environment import UnityEnvironment


if __name__ == '__main__':
    env = UnityEnvironment(file_name=None, seed=1)
    env.reset()


    # We will only consider the first Behavior
    behavior_name = list(env.behavior_specs)[0]
    print(f"Name of the behavior : {behavior_name}")
    spec = env.behavior_specs[behavior_name]


    # Examine the number of observations per Agent
    print("Number of observations : ", len(spec.observation_specs))

    # Is there a visual observation ?
    # Visual observation have 3 dimensions: Height, Width and number of channels
    vis_obs = any(len(spec.shape) == 3 for spec in spec.observation_specs)
    print("Is there a visual observation ?", vis_obs)

    # Is the Action continuous or multi-discrete ?
    if spec.action_spec.continuous_size > 0:
        print(f"There are {spec.action_spec.continuous_size} continuous actions")
    if spec.action_spec.is_discrete():
        print(f"There are {spec.action_spec.discrete_size} discrete actions")


    # How many actions are possible ?
    #print(f"There are {spec.action_size} action(s)")

    # For discrete actions only : How many different options does each action has ?
    if spec.action_spec.discrete_size > 0:
        for action, branch_size in enumerate(spec.action_spec.discrete_branches):
            print(f"Action number {action} has {branch_size} different options")



    decision_steps, terminal_steps = env.get_steps(behavior_name)

    env.set_actions(behavior_name, spec.action_spec.empty_action(len(decision_steps)))


    env.step()


    import matplotlib.pyplot as plt
    import numpy as np

    for index, obs_spec in enumerate(spec.observation_specs):
        if len(obs_spec.shape) == 3:
            print("Here is the first visual observation")
            plt.imshow(np.moveaxis(decision_steps.obs[index][0, :, :, :], 0, -1))
            plt.show()

    for index, obs_spec in enumerate(spec.observation_specs):
        if len(obs_spec.shape) == 1:
            print("First vector observations : ", decision_steps.obs[index][0,:])


    for episode in range(3):
        env.reset()
        decision_steps, terminal_steps = env.get_steps(behavior_name)
        tracked_agent = -1 # -1 indicates not yet tracking
        done = False # For the tracked_agent
        episode_rewards = 0 # For the tracked_agent
        while not done:
            # Track the first agent we see if not tracking
            # Note : len(decision_steps) = [number of agents that requested a decision]
            if tracked_agent == -1 and len(decision_steps) >= 1:
                tracked_agent = decision_steps.agent_id[0]

            # Generate an action for all agents
            action = spec.action_spec.random_action(len(decision_steps))

            # Set the actions
            env.set_actions(behavior_name, action)

            # Move the simulation forward
            env.step()

            # Get the new simulation results
            decision_steps, terminal_steps = env.get_steps(behavior_name)
            if tracked_agent in decision_steps: # The agent requested a decision
                episode_rewards += decision_steps[tracked_agent].reward
            if tracked_agent in terminal_steps: # The agent terminated its episode
                episode_rewards += terminal_steps[tracked_agent].reward
                done = True
        print(f"Total rewards for episode {episode} is {episode_rewards}")

    print(1)
