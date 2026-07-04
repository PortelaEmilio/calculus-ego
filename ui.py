"""Salida unificada por consola usando Rich.

Concentra cabeceras, mensajes de estado y tablas para que todos los módulos
del pipeline tengan el mismo aspecto y los logs queden compactos.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table


console = Console(highlight=False)


def header(title: str) -> None:
    """Separador con título centrado."""
    console.rule(f"[bold]{title}")


def info(msg: str) -> None:
    console.print(msg)


def success(msg: str) -> None:
    console.print(f"[green]{msg}")


def warn(msg: str) -> None:
    console.print(f"[yellow]{msg}")


def error(msg: str) -> None:
    console.print(f"[red]{msg}")


def kv_table(title: str, rows: Iterable[tuple[str, str]]) -> None:
    """Tabla simple de clave/valor (sin bordes ruidosos)."""
    table = Table(
        title=title or None, title_style="bold", title_justify="left",
        show_header=False, box=None, pad_edge=False,
    )
    table.add_column(style="dim", no_wrap=True, min_width=22)
    table.add_column(no_wrap=True)
    for k, v in rows:
        table.add_row(k, str(v))
    console.print(table)


def dist_table(title: str, rows: Iterable[tuple[str, int]]) -> None:
    """Tabla de distribución categoría → conteo."""
    table = Table(
        title=title or None, title_style="bold", title_justify="left",
        show_header=False, box=None, pad_edge=False,
    )
    table.add_column(style="cyan", no_wrap=True, min_width=22)
    table.add_column(justify="right", no_wrap=True, min_width=6)
    for cat, count in rows:
        table.add_row(str(cat), str(count))
    console.print(table)


def make_progress() -> Progress:
    """Barra de progreso estándar (spinner + barra + N/M + tiempos)."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    )


@contextmanager
def status(msg: str):
    """Spinner inline para operaciones cortas (carga de modelos, etc.)."""
    with console.status(f"[cyan]{msg}", spinner="dots") as st:
        yield st
