def pendulum(action):
    action = action.detach().numpy()
    action = 4 * (action - 0.5)
    return action


def cartpole(action):
    if action > 0:
        action = 1
    else:
        action = 0
    return action
