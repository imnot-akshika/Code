import argparse
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from pathlib import Path
import csv

console = Console()

def cmd_analyse(args):
    p = Path(args.file)

    if not p.exists():
        console.print(f"[red]File not found: {args.file}[/red]")
        return
    
    stat = p.stat()
    info = (
        f"File: {p.name}",
        f"Size: {stat.st_size} bytes",
        f"Extension: {p.suffix}",
        f"Last Modified: {__import__('datetime').datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M')}"
    )
    console.print(Panel("\n".join(info), title="File Info", style="cyan"))

    if p.suffix == ".csv":
        with open(p) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        console.print(f"\n[green]Rows:[/green] {len(rows)}")
        console.print(f"[green]Columns:[/green] {', '.join(rows[0].keys())}")
        
        table = Table(title="First 3 rows")
        for col in rows[0].keys():
            table.add_column(col, style="cyan")
        for row in rows[:3]:
            table.add_row(*row.values())
        console.print(table)
    
    # handle text
    elif p.suffix in (".txt", ".py", ".md", ".json"):
        text = p.read_text(encoding="utf-8")
        lines = text.splitlines()
        words = text.split()
        console.print(f"\n[green]Lines:[/green] {len(lines)}")
        console.print(f"[green]Words:[/green] {len(words)}")
        console.print(f"[green]Characters:[/green] {len(text)}")


def cmd_search(args):
        p = Path(args.directory)
        if not p.is_dir():
            console.print(f"[red]Directory not found: {args.directory}[/red]")
            return
        
        matches = [f for f in p.rglob(args.pattern) 
           if f.is_file() 
           and f.stat().st_size >= args.min_size
           and (args.max_size is None or f.stat().st_size <= args.max_size)]
        console.print(f"[green]Found {len(matches)} matches for pattern '{args.pattern}' in '{args.directory}'[/green]")
        for match in matches:
            console.print(f"- {match}")


def cmd_stats(args):
        p = Path(args.directory)
        if not p.is_dir():
            console.print(f"[red]Directory not found: {args.directory}[/red]")
            return
        
        file_counts = {}
        for file in p.rglob("*.*"):
            ext = file.suffix
            file_counts[ext] = file_counts.get(ext, 0) + 1
        
        table = Table(title="File Type Statistics")
        table.add_column("Extension", style="cyan")
        table.add_column("Count", style="magenta")
        for ext, count in file_counts.items():
            table.add_row(ext, str(count))
        
        console.print(table)

def cmd_duplicates(args):
        p = Path(args.directory)
        if not p.is_dir():
            console.print(f"[red]Directory not found: {args.directory}[/red]")
            return
        
        hash_map = {}
        duplicates = []
        for file in p.rglob("*.*"):
            file_hash = __import__('hashlib').md5(file.read_bytes()).hexdigest()
            if file_hash in hash_map:
                duplicates.append((file, hash_map[file_hash]))
            else:
                hash_map[file_hash] = file
        
        console.print(f"[green]Found {len(duplicates)} duplicate files in '{args.directory}'[/green]")
        for dup in duplicates:
            console.print(f"- {dup[0]} and {dup[1]}")


parser = argparse.ArgumentParser(description="File analysis tool")
subparsers = parser.add_subparsers(dest="command")

analyse_parser = subparsers.add_parser("analyse", help="Analyse a file")
analyse_parser.add_argument("file", help="File to analyse")

search_parser = subparsers.add_parser("search", help="Search for files")
search_parser.add_argument("directory", help="Directory to search")
search_parser.add_argument("--pattern", default="*", help="Glob pattern")
search_parser.add_argument("--min-size", type=int, default=0)
search_parser.add_argument("--max-size", type=int, default=None)

stats_parser = subparsers.add_parser("stats", help="Directory statistics")
stats_parser.add_argument("directory")

duplicates_parser = subparsers.add_parser("duplicates", help="Find duplicate files")
duplicates_parser.add_argument("directory")

args = parser.parse_args()

if args.command == "analyse":
    cmd_analyse(args)
elif args.command == "search":
    cmd_search(args)
elif args.command == "stats":
    cmd_stats(args)
elif args.command == "duplicates":
    cmd_duplicates(args)