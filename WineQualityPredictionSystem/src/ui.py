"""
ui.py
-----
Small helper module that gives the CLI its "polish": colored text, banners,
section headers, tables. Nothing here touches data or the model - it is
pure presentation, kept separate on purpose (separation of concerns) so the
rest of the codebase never has to worry about how things look on screen.

Uses colorama so the colors also work correctly on Windows terminals
(plain ANSI codes can look broken on old Windows consoles otherwise).
"""

from colorama import Fore, Style, init as colorama_init

# autoreset=True means we don't have to manually reset color after every print
colorama_init(autoreset=True)

from src import config


def banner(title: str = config.APP_TITLE) -> None:
    """Prints the big top banner, same style as the required menu."""
    line = "=" * config.DIVIDER_WIDTH
    print(Fore.CYAN + Style.BRIGHT + line)
    print(Fore.CYAN + Style.BRIGHT + title.center(config.DIVIDER_WIDTH))
    print(Fore.CYAN + Style.BRIGHT + line)


def divider() -> None:
    print(Fore.CYAN + "-" * config.DIVIDER_WIDTH)


def section(title: str) -> None:
    """Header used at the top of each menu option's screen."""
    print()
    print(Fore.YELLOW + Style.BRIGHT + f">>> {title}")
    divider()


def success(msg: str) -> None:
    print(Fore.GREEN + Style.BRIGHT + f"[OK] {msg}")


def warn(msg: str) -> None:
    print(Fore.YELLOW + Style.BRIGHT + f"[!] {msg}")


def error(msg: str) -> None:
    print(Fore.RED + Style.BRIGHT + f"[ERROR] {msg}")


def info(msg: str) -> None:
    print(Fore.WHITE + msg)


def highlight(msg: str) -> None:
    """Use for the single most important number/result on screen."""
    print(Fore.MAGENTA + Style.BRIGHT + msg)


def key_value(key: str, value, width: int = 28) -> None:
    print(Fore.WHITE + f"{key:<{width}}" + Fore.GREEN + Style.BRIGHT + f"{value}")


def pause() -> None:
    input(Fore.CYAN + "\nPress Enter to return to the menu...")


def print_table(headers, rows) -> None:
    """
    Lightweight table printer (no extra dependency like `tabulate` needed).
    `headers` is a list of column names. `rows` is a list of lists/tuples
    with values in the same order as `headers`. Columns auto-size to their
    widest cell.
    """
    headers = [str(h) for h in headers]
    str_rows = [[str(cell) for cell in row] for row in rows]

    widths = [len(h) for h in headers]
    for row in str_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def format_row(cells):
        return "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells))

    print(Fore.YELLOW + Style.BRIGHT + format_row(headers))
    print(Fore.CYAN + "  ".join("-" * w for w in widths))
    for row in str_rows:
        print(Fore.WHITE + format_row(row))
