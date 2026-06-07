import torch

def compute_gloo(elbo_not_reduced, c_sample_logprob, encoder, n_samples=None):

    """
    Estimate gradients using Reinforce LOO.
        
        Parameters
        ----------
        elbo_not_reduced : torch.tensor
            Matrix with the ELBO values obtained for the data points in the batch for a given number of samples before reducing. [shape (n_onbservations, n_samples)]
        c_sample_logprob : torch.tensor
            The log probability of the sampled words c.
        encoder: Encoder instance
            Encoder of the model.
        n_sampes : int, optional
            Number of samples used to compute the ELBO.
    
        Returns
        -------
        Tuple containing the estimated gradients of the encoder parameters.
    """

    n_obs, n_samples_dim = elbo_not_reduced.shape

    if n_samples is None or n_samples > n_samples_dim:
        n_samples = n_samples_dim

    sum_elbo_loss = torch.sum(-elbo_not_reduced, dim=1)

    grads = [0.0] * len(list(encoder.parameters())) # Preallocate with zeros

    for i in range(n_obs):
        grad_xs_total = None
        for s in range(n_samples):
            grad_xs = list(torch.autograd.grad(c_sample_logprob[s, i], encoder.parameters(), retain_graph=True))
            term1 = [g * (-elbo_not_reduced[i, s]) for g in grad_xs]
            term2 = [(-1 / n_samples) * g * sum_elbo_loss[i] for g in grad_xs]

            grad_xs_total = [(t1 + t2)/(n_samples - 1) for t1, t2 in zip(term1, term2)] if grad_xs_total is None else [gt + (t1 + t2)/(n_samples - 1) for gt, t1, t2 in zip(grad_xs_total, term1, term2)]

        # Accumulate gradients directly into grads list
        for idx, grad in enumerate(grad_xs_total):
            grads[idx] += grad
    
    return tuple(grads)