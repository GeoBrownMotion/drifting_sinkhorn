import json
from dataclasses import dataclass
from pathlib import Path


DINOV2_DIMS = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitg14": 1536,
}


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    groups: int
    dim: int


def get_config_value(conf, path: str, default=None):
    node = conf
    for part in path.split("."):
        if not hasattr(node, part):
            return default
        node = getattr(node, part)
    return node


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def format_flops(value: int | float) -> str:
    if value >= 1e12:
        return f"{value / 1e12:.3f} TFLOPs"
    if value >= 1e9:
        return f"{value / 1e9:.3f} GFLOPs"
    if value >= 1e6:
        return f"{value / 1e6:.3f} MFLOPs"
    return f"{value:.0f} FLOPs"


def dinov2_feature_groups(conf) -> int:
    groups = 16 * 16
    if bool(get_config_value(conf, "encoder.params.global_stat", True)):
        groups += 2
    if bool(get_config_value(conf, "encoder.params.patch2_stat", True)):
        groups += 2 * (16 // 2) * (16 // 2)
    if bool(get_config_value(conf, "encoder.params.patch4_stat", True)):
        groups += 2 * (16 // 4) * (16 // 4)
    return groups


def feature_specs_from_config(conf, dinov2_dim_override: int | None = None) -> list[FeatureSpec]:
    image_size = int(get_config_value(conf, "model.params.input_size", 32))
    in_channels = int(get_config_value(conf, "model.params.in_channels", 3))
    specs = [FeatureSpec(name="x", groups=1, dim=in_channels * image_size * image_size)]

    encoder_target = str(get_config_value(conf, "encoder.target", ""))
    if encoder_target.endswith("DINOv2Encoder"):
        model_name = str(get_config_value(conf, "encoder.params.model_name", "dinov2_vitb14"))
        dinov2_dim = dinov2_dim_override or DINOV2_DIMS.get(model_name)
        if dinov2_dim is None:
            raise ValueError(
                f"Unknown DINOv2 model {model_name!r}; pass dinov2_dim explicitly."
            )
        groups = dinov2_feature_groups(conf)
        for layer in list(get_config_value(conf, "encoder.params.layers", ["norm"])):
            specs.append(FeatureSpec(name=f"dinov2-{layer}", groups=groups, dim=dinov2_dim))

    return specs


def plan_normalization_flops(plan: str, elements: int, sinkhorn_iters: int) -> int:
    # Approximate constants make the O(B^2 T) Sinkhorn term explicit.
    if plan == "one-sided":
        return 5 * elements
    if plan == "two-sided":
        return 28 * elements
    if plan == "sinkhorn":
        return (20 * sinkhorn_iters + 10) * elements
    raise ValueError(f"Unknown plan: {plan}")


def estimate_feature_flops(
        spec: FeatureSpec,
        num_real: int,
        num_fake: int,
        plan: str,
        sinkhorn_iters: int,
        normalize_feature: bool,
        normalize_drift: bool,
) -> dict:
    pos_elements = num_fake * num_real
    neg_elements = num_fake * num_fake
    elements = pos_elements + neg_elements

    distance_passes = 2 if normalize_feature else 1
    distance_flops = distance_passes * 3 * elements * spec.dim
    plan_flops = plan_normalization_flops(plan, elements, sinkhorn_iters)
    barycentric_flops = 2 * elements * spec.dim
    other_flops = num_fake * spec.dim
    if normalize_drift:
        other_flops += 3 * num_fake * spec.dim

    per_group = distance_flops + plan_flops + barycentric_flops + other_flops
    total = spec.groups * per_group
    return {
        "name": spec.name,
        "groups": spec.groups,
        "dim": spec.dim,
        "distance_flops": spec.groups * distance_flops,
        "plan_flops": spec.groups * plan_flops,
        "barycentric_flops": spec.groups * barycentric_flops,
        "other_flops": spec.groups * other_flops,
        "total_flops": total,
    }


def estimate_drift_flops(
        conf,
        plans: list[str],
        num_real: int | None = None,
        num_fake: int | None = None,
        sinkhorn_iters: int | None = None,
        dinov2_dim: int | None = None,
) -> dict:
    num_real = int(num_real or conf.train.num_real_samples)
    num_fake = int(num_fake or conf.train.num_fake_samples)
    sinkhorn_iters = int(sinkhorn_iters or conf.drifting.get("sinkhorn_iters", 30))
    normalize_feature = bool(conf.drifting.get("normalize_feature", False))
    normalize_drift = bool(conf.drifting.get("normalize_drift", False))
    specs = feature_specs_from_config(conf, dinov2_dim)

    results = {
        "scope": "drifting-field computation only",
        "setting": {
            "num_real": num_real,
            "num_fake": num_fake,
            "sinkhorn_iters": sinkhorn_iters,
            "normalize_feature": normalize_feature,
            "normalize_drift": normalize_drift,
        },
        "features": [spec.__dict__ for spec in specs],
        "plans": {},
    }
    for plan in plans:
        features = [
            estimate_feature_flops(
                spec=spec,
                num_real=num_real,
                num_fake=num_fake,
                plan=plan,
                sinkhorn_iters=sinkhorn_iters,
                normalize_feature=normalize_feature,
                normalize_drift=normalize_drift,
            )
            for spec in specs
        ]
        results["plans"][plan] = {
            "features": features,
            "total_flops": sum(item["total_flops"] for item in features),
        }
    return results


def make_flops_report(results: dict) -> str:
    lines = [
        "# CIFAR-10 Per-Iteration Drift FLOPs Estimate",
        "",
        "This estimates only the drifting-field computation, not UNet/DINOv2 forward or backward.",
        "Multiply-add is counted as two FLOPs for barycentric matrix multiplies; distance and normalization constants are approximate.",
        "",
        "## Setting",
        "",
        f"- num_real: {results['setting']['num_real']}",
        f"- num_fake: {results['setting']['num_fake']}",
        f"- sinkhorn_iters: {results['setting']['sinkhorn_iters']}",
        f"- normalize_feature: {results['setting']['normalize_feature']}",
        f"- normalize_drift: {results['setting']['normalize_drift']}",
        "",
        "## Summary",
        "",
        "| plan | total drift FLOPs / iteration | overhead vs two-sided |",
        "| --- | ---: | ---: |",
    ]

    baseline = results["plans"].get("two-sided", {}).get("total_flops")
    for plan, item in results["plans"].items():
        overhead_text = "n/a"
        if baseline:
            overhead_text = f"{item['total_flops'] / baseline:.3f}x"
        lines.append(f"| {plan} | {format_flops(item['total_flops'])} | {overhead_text} |")

    lines.extend([
        "",
        "## Breakdown",
        "",
        "| plan | feature | groups | dim | distance | plan norm | barycentric matmul | total |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for plan, item in results["plans"].items():
        for feature in item["features"]:
            lines.append(
                "| "
                f"{plan} | {feature['name']} | {feature['groups']} | {feature['dim']} | "
                f"{format_flops(feature['distance_flops'])} | "
                f"{format_flops(feature['plan_flops'])} | "
                f"{format_flops(feature['barycentric_flops'])} | "
                f"{format_flops(feature['total_flops'])} |"
            )
    lines.extend([
        "",
        "## Asymptotic Form",
        "",
        "For global fake batch Bf, real batch Br, feature dimension d, feature groups g, and Sinkhorn iterations T:",
        "",
        "- pairwise distances: O(g * Bf * (Br + Bf) * d)",
        "- barycentric matmul: O(g * Bf * (Br + Bf) * d)",
        "- two-sided normalization: O(g * Bf * (Br + Bf))",
        "- Sinkhorn normalization: O(g * T * Bf * (Br + Bf))",
        "- memory for coupling logits/plans: O(Bf * (Br + Bf)) per feature group processed",
    ])
    return "\n".join(lines) + "\n"


def write_flops_files(results: dict, exp_dir: str | Path) -> None:
    exp_dir = Path(exp_dir)
    with open(exp_dir / "complexity.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(exp_dir / "complexity.md", "w") as f:
        f.write(make_flops_report(results))
