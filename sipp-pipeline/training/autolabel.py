"""Step 2: Auto-label frames with Grounding DINO via Autodistill.

Usage:
    python autolabel.py
    python autolabel.py --frames ./frames --output ./labels_autodistill
"""
import argparse
from autodistill_grounding_dino import GroundingDINO
from autodistill.detection import CaptionOntology
from config import FRAMES_DIR, AUTODISTILL_OUTPUT, BAR_ONTOLOGY


def main():
    parser = argparse.ArgumentParser(description="Auto-label bar frames with Grounding DINO")
    parser.add_argument("--frames", default=FRAMES_DIR, help="Input frames directory")
    parser.add_argument("--output", default=AUTODISTILL_OUTPUT, help="Output labels directory")
    args = parser.parse_args()

    print(f"Labeling {args.frames} -> {args.output}")
    print(f"Ontology: {len(BAR_ONTOLOGY)} prompts")

    base_model = GroundingDINO(
        ontology=CaptionOntology(BAR_ONTOLOGY)
    )

    base_model.label(
        input_folder=args.frames,
        output_folder=args.output,
    )

    print(f"Autodistill labeling complete: {args.output}")


if __name__ == "__main__":
    main()
