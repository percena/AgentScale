"""Command-line interface."""

import click


@click.command()
@click.version_option()
def main() -> None:
    """AgentScale."""


if __name__ == "__main__":
    main(prog_name="agentscale")  # pragma: no cover
