from pathlib import Path
from datetime import datetime
import shutil
import os

class FileManager:
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)

    def backup(self, source: str, backup_dir: str) -> str:
        source = Path(source)
        backup = Path(backup_dir)
        backup.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        new_name = f"{source.stem}_{timestamp}{source.suffix}"
        dest = backup / new_name

        shutil.copy2(str(source), str(dest))
        return str(dest)
    
    def find_duplicates(self, directory: str) -> dict[int, list[Path]]:
        dir = Path(directory)
        file_map = {}
        for f in dir.rglob("*"):
            if f.is_file():
                file_map.setdefault(f.stat().st_size, []).append(f)
        return {size: files for size, files in file_map.items() if len(files) > 1}
    
    def cleanup_empty_dirs(self, directory: str) -> int:
        dir = Path(directory)
        count = 0
        for d in dir.rglob("*"):
            if d.is_dir() and not any(d.iterdir()):
                d.rmdir()
                count += 1
        return count
    
    def get_largest_files(self, directory: str, n: int = 5) -> list[tuple[Path, int]]:
        dir = Path(directory)
        files = [(f, f.stat().st_size) for f in dir.rglob("*") if f.is_file()]
        files.sort(key=lambda x: x[1], reverse=True)
        return files[:n]
    
    def summary(self, directory: str) -> dict:
        dir  = Path(directory)
        total_files = sum(1 for f in dir.rglob("*") if f.is_file())
        total_dirs = sum(1 for d in dir.rglob("*") if d.is_dir())
        total_size_bytes = sum(f.stat().st_size for f in dir.rglob("*") if f.is_file())
        largest_file = self.get_largest_files(directory)
        oldest_file = min((f for f in dir.rglob("*") if f.is_file()), key=lambda x: x.stat().st_mtime, default=None)
        newest_file = max((f for f in dir.rglob("*") if f.is_file()), key=lambda x: x.stat().st_mtime, default=None)
        return {
            "total_files": total_files,
            "total_dirs": total_dirs,
            "total_size_bytes": total_size_bytes,
            "largest_file": largest_file[0] if largest_file else None,
            "oldest_file": oldest_file,
            "newest_file": newest_file
        }

# Example usage:
fm = FileManager(".")

# create some test files first
Path("test_dir").mkdir(exist_ok=True)
Path("test_dir/file1.txt").write_text("hello world")
Path("test_dir/file2.txt").write_text("hello world")  # same size — duplicate
Path("test_dir/file3.py").write_text("print('hi')")
Path("test_dir/empty_sub").mkdir(exist_ok=True)

print(fm.summary("test_dir"))
print(fm.find_duplicates("test_dir"))
print(fm.get_largest_files("test_dir", 3))
print(fm.cleanup_empty_dirs("test_dir"))