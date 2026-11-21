"""Prepare local Q1, Bumi and G1 BONES-SEED/LAFAN1 motions at 50 Hz."""
import argparse
import os
from pathlib import Path

from morphmimic.data.local_prepare import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New output directory; never overwrites old data')
    parser.add_argument('--limit', type=int, help='Maximum clips per robot and dataset for smoke conversion')
    parser.add_argument('--workers', type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument('--resume', action='store_true', help='Verify source/asset/output hashes and reuse completed clips')
    args = parser.parse_args()
    result = prepare(args.output, limit=args.limit, workers=args.workers, resume=args.resume)
    print(f'Prepared {len(result["clips"])} clips; {len(result["retargeted_bumi_lafan1"])} Bumi LAFAN1 clips retargeted.')


if __name__ == '__main__':
    main()
