"""Console demo for tripod gait stepping and IK solve summaries."""

from hexapod_ik.gait.tripod_gait import run_tripod_gait_demo


def main():
    print("Running tripod gait demo (console summary)...")
    result = run_tripod_gait_demo(cycles=1, verbose=False)
    print(f"num_steps={result['num_steps']}")
    print(f"num_frames={len(result['angle_frames'])}")
    print(f"legs={result['leg_order']}")


if __name__ == "__main__":
    main()
