"""
OpenMedImgDataVal: Medical Image Data Validation Pipeline

Usage:
    python src/main.py --modality xray --task class --dv shap
    python src/main.py --modality xray --task class --dv ot
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    parser = argparse.ArgumentParser(description="OpenMedImgDataVal: Medical Image Data Validation Pipeline")
    parser.add_argument(
        "--modality",
        type=str,
        required=True,
        choices=["xray", "mri"],
        help="Imaging modality: xray or mri",
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["seg", "class"],
        help="Task type: seg (segmentation) or class (classification)",
    )
    parser.add_argument(
        "--dv",
        type=str,
        required=True,
        choices=["shap", "ot"],
        help="Data valuation method: shap (KNN-Shapley) or ot (optimal transport)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.out_dir = os.path.join("output", args.modality, args.task, args.dv)

    print("=" * 60)
    print("OpenMedImgDataVal Pipeline")
    print("=" * 60)
    print(f"  Modality : {args.modality}")
    print(f"  Task     : {args.task}")
    print(f"  DV Method: {args.dv}")
    print(f"  Output   : {args.out_dir}")
    print("=" * 60)

    if args.modality == "xray" and args.task == "class":
        _run_xray_classification(args)
    else:
        print(f"\nERROR: Pipeline for --modality {args.modality} --task {args.task} is not yet implemented.")
        print("Currently supported: --modality xray --task class")
        sys.exit(1)


def _run_xray_classification(args):
    from xray_class.pipeline import run_pipeline

    class _LazyMethodParams:
        """Deferred so defaults are applied by the pipeline before formatting."""

        def __init__(self, args):
            self._args = args

        def __str__(self):
            a = self._args
            if a.dv == "ot":
                return f"ot_reg: {a.ot_reg}"
            lines = [
                f"shapley_mstar: {a.shapley_mstar}",
                f"shapley_batch_val: {a.shapley_batch_val}",
                f"k_candidates: {a.k_candidates}",
            ]
            out = "\n".join(lines)
            if getattr(a, "_optimized_k", None) is not None:
                out += f"\noptimized_k (selected): {a._optimized_k}"
                if a._k_results:
                    out += "\nk_optimization_results:"
                    for kv, acc in sorted(a._k_results.items()):
                        out += f"\n  k={kv}: val_acc={acc:.6f}"
            return out

    if args.dv == "ot":
        from dv.ot.sinkhorn import compute_ot_values

        run_pipeline(args, compute_ot_values, "OT", _LazyMethodParams(args))

    elif args.dv == "shap":
        from dv.shap.knn_shapley import compute_shapley_values

        run_pipeline(args, compute_shapley_values, "Shapley", _LazyMethodParams(args))


if __name__ == "__main__":
    main()
