def get_triangular_lr(iteration, step_size, base_lr, max_lr):
    """Given the inputs, calculates the lr that should be applicable for this iteration"""
    """http://teleported.in/posts/cyclic-learning-rate/"""
    bump = float(max_lr - base_lr)/float(step_size)
    cycle = iteration % (2 * step_size)
    if cycle < step_size:
        lr = base_lr + cycle*bump
    else:
        lr = max_lr - (cycle - step_size) * bump
    return lr
