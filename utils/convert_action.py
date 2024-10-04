def pendulum(action):
    action = action.detach().numpy()
    action = 4 * (action - 0.5)
    return action


def cartpole(action):
    return 1 if action > 0 else 0


def cartpole2(action):
    action = int(action.item())
    return action
