def get_triangular_lr(iteration, stepsize, base_lr, max_lr):
    """Given the inputs, calculates the lr that should be applicable for this iteration"""
    """http://teleported.in/posts/cyclic-learning-rate/"""
    bump = float(max_lr - base_lr)/float(stepsize)
    cycle = iteration%(2*stepsize)
    if cycle < stepsize:
        lr = base_lr + cycle*bump
    else:
        lr = max_lr - (cycle-stepsize)*bump
    return lr
