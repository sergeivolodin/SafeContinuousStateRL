import numpy as np

def cost_cartpole_left(obs):
    """ Calculate scalar cost of one observation """
    assert isinstance(obs, np.ndarray) and obs.shape == (4,), "Input must be an np-array [x xdot phi phidot]"
    
    # parsing input
    x, x_dot, phi, phi_dot = obs
    
    #X_MAX = 1.0
    #X_DOT_MAX = 0.5
    #PHI_MAX = 0.1
    #PHI_DOT_MAX = 0.5
    
    #if x < 0 or phi < 0:
    #    return 1
    if x > 0: return 1
    
    #if np.any(np.abs([x, x_dot, phi, phi_dot]) > [X_MAX, X_DOT_MAX, PHI_MAX, PHI_DOT_MAX]):
    #    return 1
    
    # in all other cases no cost
    return 0

