"""MeanFlow loss for a short video latent.

u is the average velocity. The training target is v - (t - r) du/dt, and the
target is stop-gradient. r = t on most samples, which reduces to flow matching.
"""

from __future__ import annotations

import torch


def sample_tr(batch: int, device: torch.device, data_proportion: float):
    def logit_normal(n):
        return torch.sigmoid(torch.randn(n, device=device) * 1.0 - 0.4)

    t = logit_normal(batch)
    r = logit_normal(batch)
    t, r = torch.maximum(t, r), torch.minimum(t, r)
    same = torch.rand(batch, device=device) < data_proportion
    r = torch.where(same, t, r)
    return t, r


def meanflow_loss(model, x, vision, view_id, data_proportion: float = 0.75):
    batch = x.shape[0]
    t, r = sample_tr(batch, x.device, data_proportion)
    epsilon = torch.randn_like(x)
    t_b = t.view(batch, 1, 1, 1, 1)
    r_b = r.view(batch, 1, 1, 1, 1)
    z = (1.0 - t_b) * x + t_b * epsilon
    v = epsilon - x
    vision = vision.detach()
    u = model(z, t, r, vision, view_id)

    def velocity(z_, t_, r_):
        return model(z_, t_, r_, vision, view_id)

    with torch.no_grad():
        _, dudt = torch.func.jvp(
            velocity,
            (z.detach(), t.detach(), r.detach()),
            (v.detach(), torch.ones_like(t), torch.zeros_like(r)),
        )
    target = (v - (t_b - r_b).clamp(0.0, 1.0) * dudt).detach()
    per_sample = (u - target).flatten(1).pow(2).mean(1)
    loss = (per_sample / (per_sample.detach() + 1e-2)).mean()
    return loss, per_sample.mean().detach()


def endpoint_loss(model, x, vision, view_id):
    """Supervise the exact one-step map used at inference.

    On the linear path the average velocity is ε − x, so z_0 = ε − u(ε, 0, 1).
    """
    batch = x.shape[0]
    epsilon = torch.randn_like(x)
    t = torch.ones(batch, device=x.device)
    r = torch.zeros(batch, device=x.device)
    u = model(epsilon, t, r, vision.detach(), view_id)
    pred = epsilon - u
    per_sample = (pred - x).flatten(1).pow(2).mean(1)
    return per_sample.mean(), per_sample.mean().detach(), pred
