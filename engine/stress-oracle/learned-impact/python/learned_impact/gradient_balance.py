"""Experimental two-objective projection on shared parameters only.

Primary = fracture BCE, auxiliary = every other weighted training objective.
If their raw gradients conflict, remove the auxiliary component opposing the
primary gradient. Task-specific heads retain their full scalar-loss gradients.
This is a training update rule, not a new physical constraint or guarantee of
improvement: Adam's preconditioning, momentum and finite steps can change the
actual update direction. It does not alter inference or oracle accuracy settings.
"""
import torch

HEAD_PREFIXES=('break_head.','time_head.','bond_util.','chunk_delta.','joint_end.','force_end.','world_motion.')


def shared_parameters(model):
    return [p for name,p in model.named_parameters() if p.requires_grad and not name.startswith(HEAD_PREFIXES)]


def projected_backward(total_loss,primary_loss,parameters):
    """Call after zero_grad; retain ordinary gradients on all other parameters."""
    parameters=list(parameters)
    if not parameters:raise ValueError('shared parameters required')
    if any(p.grad is not None for p in parameters):raise ValueError('zero_grad(set_to_none=True) before projected backward')
    primary=torch.autograd.grad(primary_loss,parameters,retain_graph=True,allow_unused=True)
    total_loss.backward()
    reference=primary_loss.detach()
    denominator=reference.new_zeros(());dot=reference.new_zeros(())
    for p,g in zip(parameters,primary):
        if g is None:continue
        denominator+=g.square().sum()
        dot+=((p.grad-g)*g).sum() if p.grad is not None else -g.square().sum()
    if not torch.isfinite(denominator) or not torch.isfinite(dot):raise ValueError('nonfinite shared gradient')
    safe_denominator=torch.where(denominator>0,denominator,torch.ones_like(denominator))
    coefficient=torch.where(denominator>0,(-dot).clamp_min(0)/safe_denominator,reference.new_zeros(()))
    with torch.no_grad():
        for p,g in zip(parameters,primary):
            if g is None:continue
            if p.grad is None:p.grad=coefficient*g
            else:p.grad.add_(coefficient*g)
    return {'projected':bool(dot<0 and denominator>0),'auxiliary_dot_primary_before':float(dot),
            'primary_norm_squared':float(denominator),'projection_coefficient':float(coefficient)}
