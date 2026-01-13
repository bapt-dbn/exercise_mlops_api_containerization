import typer

from src.train.train import run_training


def main(
    config: str = typer.Option(
        "config/config.yaml",
        "--config",
        "-c",
        help="Path to configuration file",
    ),
) -> None:
    run_training(config)


if __name__ == "__main__":
    typer.run(main)
