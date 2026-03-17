import argparse
import os


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
        help="Data valuation method: shap or ot (optimal transport)",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to input data directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/",
        help="Path to output directory (default: output/)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("OpenMedImgDataVal Pipeline")
    print("=" * 60)

    # Step 1: Parse and display configuration
    print("\n[Config]")
    print(f"  Modality : {args.modality}")
    print(f"  Task     : {args.task}")
    print(f"  DV Method: {args.dv}")
    print(f"  Input    : {args.input or '(not specified — load data from default location)'}")
    print(f"  Output   : {args.output}")

    # Step 2: Resolve input data directory
    # If --input is provided, load data from that directory.
    # Otherwise, data should be placed in: data/<modality>/<task>/
    if args.input:
        input_dir = args.input
    else:
        input_dir = os.path.join("data", args.modality, args.task)
    print(f"\n[Step 1] Load data from: {input_dir}")

    # Step 3: Ensure output directory exists
    output_dir = args.output
    print(f"[Step 2] Output will be saved to: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # Step 4: Load model for the given modality + task
    if args.modality == "xray":
        print("[Step 3] Loading X-ray image preprocessing pipeline")
        if args.task == "seg":
            print("  → Loading X-ray segmentation model (e.g., U-Net for lung/bone segmentation)")
        elif args.task == "class":
            print("  → Loading X-ray classification model (e.g., DenseNet for pathology detection)")
    elif args.modality == "mri":
        print("[Step 3] Loading MRI volume preprocessing pipeline")
        if args.task == "seg":
            print("  → Loading MRI segmentation model (e.g., 3D U-Net for organ/tumor segmentation)")
        elif args.task == "class":
            print("  → Loading MRI classification model (e.g., ResNet for disease classification)")

    # Step 5: Run data valuation
    if args.dv == "shap":
        print("[Step 4] Running SHAP-based data valuation")
        print("  → Computing Shapley values for each training sample")
        print("  → Estimating marginal contribution via Monte Carlo sampling")
    elif args.dv == "ot":
        print("[Step 4] Running optimal transport data valuation")
        print("  → Computing Wasserstein distances between data distributions")
        print("  → Estimating sample transport costs")

    # Step 6: Save results
    print(f"[Step 5] Save results to {output_dir}")
    if args.dv == "shap":
        print(f"  → Writing Shapley value scores to {output_dir}")
    elif args.dv == "ot":
        print(f"  → Writing transport cost matrix to {output_dir}")

    print("\n" + "=" * 60)
    print("Pipeline complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
