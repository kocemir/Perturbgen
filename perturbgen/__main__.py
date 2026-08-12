"""Command-line interface for PerturbGen."""

import os

# Headless-first: kill X11 "No protocol specified" before any heavy import.
# OpenMPI/hwloc (via mpi4py) is the usual culprit on this host — disable GL probe.
os.environ.pop("DISPLAY", None)
os.environ.pop("WAYLAND_DISPLAY", None)
os.environ["MPLBACKEND"] = "Agg"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("WANDB_DISABLE_CODE", "true")
os.environ.setdefault("WANDB_CONSOLE", "off")
os.environ.setdefault("HWLOC_COMPONENTS", "-gl")
os.environ.setdefault("HWLOC_GL_LINUX_NVIDIA_DISABLE", "1")

import click


class OrderedGroup(click.Group):
    """`click.Group` which prints its subcommands in a specific order.

    By default, Click will show subcommands in alphabetical order.
    Sometimes it makes more sense to use a different order, which you
    can manually specify by using this class.

    Usage:

        @click.group(
            cls=OrderedGroup,
            order=["foo", "bar"],
        )
        def main(): ...

        @main.command()
        def foo(): ...

        @main.command()
        def bar(): ...
    """
    def __init__(self, *args, order, **kwargs):
        super().__init__(*args, **kwargs)
        self.__order = order

    def list_commands(self, ctx):
        all_names = super().list_commands(ctx)
        all_names_set = set(all_names)
        ordered_names = [name for name in self.__order if name in all_names_set]
        other_names = [name for name in all_names if name not in self.__order]
        return ordered_names + other_names


@click.group(
    cls=OrderedGroup,
    order=[
        "tokenise",
        "train-mask",
        "train-decoder",
        "train-jepa",
        "train-jepa-gene-query",
        "train-jepa-decoder",
        "extract-embedding",
        "eval-jepa",
    ],
)
def main():
    pass


@main.command(context_settings={"ignore_unknown_options": True, "help_option_names": []})
@click.argument("args", nargs=-1)
def tokenise(args):
    """Data preprocessing, tokenisation"""
    click.echo("loading, please wait...")
    from perturbgen.pp.GF_tokenisation import main
    main(args)


@main.command(context_settings={"ignore_unknown_options": True, "help_option_names": []}, hidden=True)
@click.argument("args", nargs=-1)
def tokenize(args):
    return tokenise(args)


@main.command(context_settings={"ignore_unknown_options": True, "help_option_names": []})
@click.argument("args", nargs=-1)
def train_mask(args):
    """Training the masking model"""
    click.echo("loading, please wait...")
    from perturbgen.train import main
    main(args)


@main.command(context_settings={"ignore_unknown_options": True, "help_option_names": []})
@click.argument("args", nargs=-1)
def train_decoder(args):
    """Training the count decoder model"""
    click.echo("loading, please wait...")
    from perturbgen.train import main
    main(args)


@main.command(context_settings={"ignore_unknown_options": True, "help_option_names": []})
@click.argument("args", nargs=-1)
def extract_embedding(args):
    """Load checkpoint and extract the embeddings."""
    # Re-clear DISPLAY right before val import (some shells re-export it).
    os.environ.pop("DISPLAY", None)
    os.environ.pop("WAYLAND_DISPLAY", None)
    os.environ["MPLBACKEND"] = "Agg"
    click.echo("loading, please wait...")
    from perturbgen.val import main
    main(args)


@main.command(
    context_settings={"ignore_unknown_options": True, "help_option_names": []},
    name="train-jepa",
)
@click.argument("args", nargs=-1)
def train_jepa(args):
    """Train cell-trajectory JEPA (Phase A)."""
    click.echo("loading, please wait...")
    from perturbgen.train import main

    # Ensure train_mode=jepa unless the user already set it.
    forwarded = list(args)
    if "--train_mode" not in forwarded:
        forwarded = ["--train_mode", "jepa", *forwarded]
    main(forwarded)


@main.command(
    context_settings={"ignore_unknown_options": True, "help_option_names": []},
    name="train-jepa-gene-query",
)
@click.argument("args", nargs=-1)
def train_jepa_gene_query(args):
    """Train Gene-Query JEPA (predict per-gene target embeddings)."""
    click.echo("loading, please wait...")
    from perturbgen.train import main

    forwarded = list(args)
    if "--train_mode" not in forwarded:
        forwarded = ["--train_mode", "jepa_gene_query", *forwarded]
    main(forwarded)


@main.command(
    context_settings={"ignore_unknown_options": True, "help_option_names": []},
    name="train-jepa-decoder",
)
@click.argument("args", nargs=-1)
def train_jepa_decoder(args):
    """Train JEPA count decoder (Phase D)."""
    click.echo("loading, please wait...")
    from perturbgen.train import main

    forwarded = list(args)
    if "--train_mode" not in forwarded:
        forwarded = ["--train_mode", "jepa_decoder", *forwarded]
    main(forwarded)


if __name__ == "__main__":
    main()
