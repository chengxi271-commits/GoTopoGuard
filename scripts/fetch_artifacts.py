"""Download the versioned PRICAI measurements; no engines or models are downloaded."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,help='Use an already downloaded evidence ZIP')
    parser.add_argument('--destination',type=Path,default=ROOT/'data/pricai2026')
    args=parser.parse_args()
    manifest=json.loads((ROOT/'artifacts/evidence_manifest.json').read_text())
    archive=args.archive or ROOT/'data'/manifest['asset']
    if not archive.exists():
        if args.archive:parser.error('The supplied archive does not exist.')
        archive.parent.mkdir(parents=True,exist_ok=True)
        print(f"Downloading {manifest['bytes']/1e6:.1f} MB of recorded measurements")
        urllib.request.urlretrieve(manifest['url'],archive)
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=manifest['sha256']:
        parser.error('Archive checksum differs from the published input version; use the v0.2.0 release asset.')
    args.destination.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        destination=args.destination.resolve()
        for item in z.infolist():
            if not (destination/item.filename).resolve().is_relative_to(destination):
                parser.error('Archive entry is outside the destination.')
        z.extractall(destination)
    print(f'Inputs ready: {args.destination}')

if __name__=='__main__':
    main()
