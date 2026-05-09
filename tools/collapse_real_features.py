"""
Re-run the two-sided / sinkhorn plan-collapse diagnostic, but on REAL DINOv2
features extracted from CIFAR-10 (not random gaussian).

If two-sided actually collapses in our real-feature setup, then sinkhorn
should win — and we have a hyperparameter / training-pipeline issue, not a
fundamental "no-mechanism-here" problem.
If two-sided does NOT collapse on real features (despite CV~18%), then the
paper claim simply doesn't transfer to our pipeline for some other reason.
"""
import math
import torch
from omegaconf import OmegaConf
from utils.misc import set_seed, instantiate_from_config
from drifting_split import _two_sided_plan, _sinkhorn_plan


def main():
    set_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    conf = OmegaConf.load("configs/cifar10-unc-split-screening.yaml")
    encoder = instantiate_from_config(conf.encoder).to(device).eval()

    dataset = instantiate_from_config(conf.data)
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(0))[:1024]
    images = torch.stack([dataset[i.item()]["image"] for i in indices], dim=0).to(device)

    with torch.no_grad():
        from models.autoencoders.identity import IdentityAutoencoder
        ae = IdentityAutoencoder().to(device).eval()
        feat = encoder(images, autoencoder=ae)

    print(f"{'stream':<22} {'normalize':>9} {'eps':>8}  {'plan':>10}  {'H_row':>7} {'max_row':>8} {'||V||':>9} {'zero_rows%':>10}")
    print("-" * 100)

    for name, f in feat.items():
        F, N, D = f.shape
        # Use the first feature index. Take half as "x_fake", half as "y_pos".
        all_x = f[0].float().cpu()
        n = all_x.shape[0] // 2
        x_fake = all_x[:n]   # (n, D)
        y_pos = all_x[n:n + n]  # (n, D)
        d_raw = torch.cdist(x_fake, y_pos)

        for normalize in [True, False]:
            d = d_raw.clone()
            if normalize:
                d = d / d.mean().clamp_min(1e-3)

            for eps in [0.5, 0.2, 0.1, 0.05, 0.03, 0.02, 0.01, 0.007, 0.005, 0.003, 0.001]:
                logit_base = (-d.pow(2) / eps).unsqueeze(0).to(torch.bfloat16)

                lp = logit_base.clone()
                P_ts = _two_sided_plan(lp).float()
                # zero rows on row sum
                pre = (logit_base.float() - 0.5 * (
                    torch.logsumexp(logit_base.float(), dim=-1, keepdim=True)
                    + torch.logsumexp(logit_base.float(), dim=-2, keepdim=True)
                )).exp()
                row_sum_pre = pre.sum(dim=-1)
                frac_zero = (row_sum_pre < 1e-6).float().mean().item()

                ls = logit_base.clone()
                P_sh = _sinkhorn_plan(ls, iters=20).float()

                H_ts = -(P_ts.clamp_min(1e-30) * P_ts.clamp_min(1e-30).log()).sum(dim=-1).mean().item()
                M_ts = P_ts.max(dim=-1).values.mean().item()
                V_ts = (P_ts @ y_pos.unsqueeze(0)).norm(dim=-1).mean().item()

                H_sh = -(P_sh.clamp_min(1e-30) * P_sh.clamp_min(1e-30).log()).sum(dim=-1).mean().item()
                M_sh = P_sh.max(dim=-1).values.mean().item()
                V_sh = (P_sh @ y_pos.unsqueeze(0)).norm(dim=-1).mean().item()

                ntag = "yes" if normalize else "no"
                print(f"{name:<22} {ntag:>9} {eps:>8.4f}  {'two-sided':>10}  {H_ts:>7.3f} {M_ts:>8.3f} {V_ts:>9.3f} {frac_zero*100:>9.2f}%")
                print(f"{name:<22} {ntag:>9} {eps:>8.4f}  {'sinkhorn':>10}  {H_sh:>7.3f} {M_sh:>8.3f} {V_sh:>9.3f} {'-':>10}")


if __name__ == "__main__":
    main()
