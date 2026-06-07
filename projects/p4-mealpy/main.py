import sys
from pathlib import Path

src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))

from p4_mealpy.cli import main

if __name__ == "__main__":
    main()