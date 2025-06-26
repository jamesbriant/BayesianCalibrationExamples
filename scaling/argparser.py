import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Run the MCMC posterior sampling.")
    parser.add_argument("W", type=int, help="Number of warm-up iterations.")
    parser.add_argument("N", type=int, help="Number of main iterations.")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed.")
    parser.add_argument("--n_chain", type=int, default=2, help="Number of MCMC chains.")
    parser.add_argument(
        "--n_processes", type=int, default=1, help="Number of processes to use."
    )
    parser.add_argument(
        "--max_tree_depth",
        type=int,
        default=5,
        help="Maximum tree depth for the sampler.",
    )
    return parser.parse_args()
