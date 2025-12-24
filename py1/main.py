import argparse

from py1.runner import run_experiment


def main(k: int = 3, visualize: bool = True) -> None:
    """
    程序主入口，运行KNN实验。

    """
    run_experiment(k=k, visualize=visualize)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行KNN实验")
    parser.add_argument("--k", type=int, default=3, help="KNN的近邻数")
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="运行时不展示 seaborn 可视化",
    )
    args = parser.parse_args()
    main(k=args.k, visualize=not args.no_plot)
