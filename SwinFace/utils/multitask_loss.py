"""
Scale-normalized multi-task loss for Expression / Pose / Landmarks.

Problem it solves:
    Raw task losses live on very different numeric scales (e.g. cross-entropy
    ~1.0-2.0 vs. landmark MSE ~0.01), so a manually chosen priority weight
    (e.g. 1 / 2 / 4) doesn't actually control each task's share of the
    gradient -- the task with the largest raw magnitude dominates regardless
    of its weight.

How it works:
    For each task, we keep a running EMA of |loss| as an estimate of that
    task's "natural scale". We divide the current loss by this scale before
    applying your priority weight, so every task contributes on a comparable
    order of magnitude, and the priority weight does what you intend.

Usage:
    normalizer = ScaleNormalizedMultiTaskLoss(
        task_names=["expression", "pose", "landmarks"],
        priority_weights=[1.0, 2.0, 4.0],
    )

    # inside training loop, after computing raw per-task losses:
    losses = {
        "expression": expression_loss,   # scalar tensor
        "pose": pose_loss,               # scalar tensor
        "landmarks": landmarks_loss,     # scalar tensor
    }
    total_loss, logs = normalizer(losses)
    total_loss.backward()

    # logs is a plain dict of floats for logging/tensorboard, e.g.:
    #   {"expression/raw": 1.37, "expression/scale": 1.35, "expression/weighted": 1.02,
    #    "pose/raw": 0.29, ..., "total": 2.01}
"""

import torch
import torch.nn as nn


class ScaleNormalizedMultiTaskLoss(nn.Module):
    def __init__(self, task_names, priority_weights, ema_momentum=0.99, eps=1e-8,
                 warmup_steps=50):
        """
        Args:
            task_names: list of str, e.g. ["expression", "pose", "landmarks"]
            priority_weights: list of float, same order/length as task_names.
                These are YOUR intended priorities (e.g. [1, 2, 4]) -- after
                scale normalization, they will actually behave as priorities.
            ema_momentum: how slowly the running scale estimate updates.
                Closer to 1.0 = smoother/slower to react, closer to 0.0 =
                more reactive to recent losses. 0.99 is a reasonable default.
            eps: numerical stability floor for the scale estimate.
            warmup_steps: for the first N steps, use the *raw* running mean
                instead of EMA, since EMA is noisy/biased when initialized
                from a single sample. Prevents an unlucky first batch from
                setting a bad scale for the rest of training.
        """
        super().__init__()
        assert len(task_names) == len(priority_weights)
        self.task_names = list(task_names)
        self.priority_weights = {
            name: float(w) for name, w in zip(task_names, priority_weights)
        }
        self.momentum = ema_momentum
        self.eps = eps
        self.warmup_steps = warmup_steps

        # Running scale estimate per task (plain floats, not parameters --
        # we are NOT learning these, just tracking a moving average).
        self._scale = {name: None for name in self.task_names}
        self._step_count = {name: 0 for name in self.task_names}

    @torch.no_grad()
    def _update_scale(self, name, loss_value):
        loss_value = float(loss_value)
        self._step_count[name] += 1
        n = self._step_count[name]

        if self._scale[name] is None:
            self._scale[name] = loss_value
            return

        if n <= self.warmup_steps:
            # plain running mean during warmup (more stable than EMA on few samples)
            self._scale[name] += (loss_value - self._scale[name]) / n
        else:
            m = self.momentum
            self._scale[name] = m * self._scale[name] + (1 - m) * loss_value

    def forward(self, losses: dict):
        """
        Args:
            losses: dict mapping task_name -> scalar loss tensor (with grad).
                    Must contain exactly the tasks passed to __init__.

        Returns:
            total_loss: scalar tensor, ready for .backward()
        """
        assert set(losses.keys()) == set(self.task_names), \
            f"Expected losses for {self.task_names}, got {list(losses.keys())}"

        weighted_terms = []

        for name in self.task_names:
            raw_loss = losses[name]

            # update the running scale using the *value* (no grad needed for this)
            self._update_scale(name, raw_loss.detach())

            scale = max(self._scale[name], self.eps)
            normalized = raw_loss / scale
            weighted = normalized * self.priority_weights[name]
            weighted_terms.append(weighted)

        total_loss = torch.stack(weighted_terms).sum()
        return total_loss, weighted_terms


# ---------------------------------------------------------------------------
# Alternative: learnable uncertainty weighting (Kendall, Gal & Cipolla, 2018
# "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry
# and Semantics"). Instead of an EMA scale, this learns a log-variance per
# task jointly with the network, and naturally down-weights noisy/harder
# tasks. More "principled" but also more moving parts (extra parameters to
# add to your optimizer, and it can occasionally drive a task's weight to
# collapse if that task's loss is very noisy early on). Left here as a
# reference if you want to try it after the EMA version:
#
# class UncertaintyWeightedLoss(nn.Module):
#     def __init__(self, task_names):
#         super().__init__()
#         self.task_names = task_names
#         # log(sigma^2) per task, learned -- init at 0 means sigma^2 = 1
#         self.log_vars = nn.Parameter(torch.zeros(len(task_names)))
#
#     def forward(self, losses: dict):
#         total = 0.0
#         logs = {}
#         for i, name in enumerate(self.task_names):
#             precision = torch.exp(-self.log_vars[i])
#             term = precision * losses[name] + self.log_vars[i]
#             total = total + term
#             logs[f"{name}/log_var"] = float(self.log_vars[i].detach())
#         return total, logs
#
# NOTE: self.log_vars must be added to your optimizer's parameter list
# (e.g. optimizer.add_param_group({"params": uncertainty_loss.parameters()})).
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Integration example with an existing AdamW + scheduler training loop.
# The normalizer has NO learnable parameters, so it does NOT get added to
# `opt` -- it just sits between "compute raw task losses" and ".backward()".
# ---------------------------------------------------------------------------
#
# opt = torch.optim.AdamW(
#     params=[{"params": model.module.backbone.parameters(), 'lr': cfg.lr / 10},
#             {"params": model.module.fam.parameters()},
#             {"params": model.module.tss.parameters()},
#             {"params": model.module.om.parameters()},
#             ],
#     lr=cfg.lr, weight_decay=cfg.weight_decay)
#
# lr_scheduler = build_scheduler(
#     optimizer=opt,
#     lr_name=cfg.lr_name,
#     warmup_lr=cfg.warmup_lr,
#     min_lr=cfg.min_lr,
#     num_steps=cfg.total_step,
#     warmup_steps=cfg.warmup_step)
#
# # instantiate ONCE, outside the training loop, alongside opt/lr_scheduler
# loss_normalizer = ScaleNormalizedMultiTaskLoss(
#     task_names=["expression", "pose", "landmarks"],
#     priority_weights=[1.0, 2.0, 4.0],   # your existing intended priorities
#     ema_momentum=0.99,
# )
#
# global_step = 0
# for epoch in range(num_epochs):
#     for batch in dataloader:
#         opt.zero_grad()
#
#         expr_loss = compute_expression_loss(...)     # your existing code
#         pose_loss = compute_pose_loss(...)            # your existing code
#         landmark_loss = compute_landmarks_loss(...)   # your existing code
#
#         total_loss, logs = loss_normalizer({
#             "expression": expr_loss,
#             "pose": pose_loss,
#             "landmarks": landmark_loss,
#         })
#
#         total_loss.backward()
#         opt.step()
#         lr_scheduler.step_update(global_step)   # or lr_scheduler.step(), depending on your build_scheduler API
#         global_step += 1
#
#         if global_step % log_interval == 0:
#             log_str = "  ".join(f"{k} {v:.4f}" for k, v in logs.items())
#             logger.info(f"Step {global_step}: {log_str}")
#
# ---------------------------------------------------------------------------
# If you separately want SWA/EMA of MODEL WEIGHTS (a different, complementary
# technique -- averaging checkpoints for a smoother final model), that's
# where torch.optim.swa_utils comes in. It plugs in like this, independently
# of everything above:
#
# from torch.optim.swa_utils import AveragedModel, SWALR
#
# ema_model = AveragedModel(
#     model,
#     avg_fn=lambda avg_p, p, n: 0.999 * avg_p + 0.001 * p  # decay=0.999 EMA
# )
#
# # after each opt.step() in the loop above, also do:
# ema_model.update_parameters(model)
#
# # NOTE: if your model uses BatchNorm, call this once after training to fix
# # up BN running stats for the averaged weights before evaluating with it:
# # torch.optim.swa_utils.update_bn(dataloader, ema_model)
#
# # then evaluate/save ema_model.module instead of (or alongside) model
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    # Quick smoke test using numbers from your actual training run
    normalizer = ScaleNormalizedMultiTaskLoss(
        task_names=["expression", "pose", "landmarks"],
        priority_weights=[1.0, 2.0, 4.0],
    )

    losses = {
        "expression": torch.tensor(1.3752, requires_grad=True),
        "pose": torch.tensor(0.2903, requires_grad=True),
        "landmarks": torch.tensor(0.0114, requires_grad=True),
    }

    total_loss, logs = normalizer(losses)
    print("Step 1 (scale = current loss, so normalized = 1.0 for all tasks):")
    for k, v in logs.items():
        print(f"  {k}: {v:.4f}")

    # Simulate a later step where landmarks loss has dropped a lot
    losses2 = {
        "expression": torch.tensor(1.20, requires_grad=True),
        "pose": torch.tensor(0.25, requires_grad=True),
        "landmarks": torch.tensor(0.004, requires_grad=True),
    }
    total_loss2, logs2 = normalizer(losses2)
    print("\nStep 2 (landmarks scale hasn't caught up yet, so normalized > 1):")
    for k, v in logs2.items():
        print(f"  {k}: {v:.4f}")