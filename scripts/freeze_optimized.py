"""Write a no-overwrite reproducibility receipt for a prepared training run."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from morphmimic.train.run_identity import ManifestIdentity,create_receipt,write_receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data-manifest','source-root','config','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--image-id',required=True,help='Immutable Docker image ID sha256:<digest>')
    a=p.parse_args()
    identity=ManifestIdentity(a.data_manifest)
    config=json.loads(a.config.read_text())
    receipt=create_receipt(a.source_root,a.image_id,config,identity.sha256)
    write_receipt(a.output,receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k!='source_files'},indent=2))

if __name__=='__main__':main()
