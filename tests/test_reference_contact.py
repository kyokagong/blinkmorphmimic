import hashlib
import json

import pytest
import torch


def test_reference_floor_binding_preserves_subset_order_and_unknowns(tmp_path):
    from morphmimic.train.reference_contact import load_reference_floors
    rows=[dict(robot='q1', name=n, output=str(tmp_path/f'{n}.npz'), output_sha256=n*64)
          for n in ('a','b')]
    manifest=tmp_path/'manifest.json'; manifest.write_text(json.dumps(dict(clips=rows)))
    receipt=tmp_path/'contact-floor.json'
    data=dict(schema_version=1,prepared_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        clips=[dict(robot='q1', name=r['name'],path=r['output'],sha256=r['output_sha256'],
            offset_m=.07 if i==0 else 0.,valid=i==0) for i,r in enumerate(rows)])
    receipt.write_text(json.dumps(data))
    floor, valid, digest=load_reference_floors(receipt,manifest,'q1',[rows[1]['output'],rows[0]['output']],'cpu')
    torch.testing.assert_close(floor,torch.tensor([0.,.07]))
    assert valid.tolist()==[False,True]
    assert digest==hashlib.sha256(receipt.read_bytes()).hexdigest()
    data['clips'][0]['sha256']='f'*64; receipt.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='source'):
        load_reference_floors(receipt,manifest,'q1',[rows[0]['output']],'cpu')


def test_reference_floor_rejects_stale_manifest(tmp_path):
    from morphmimic.train.reference_contact import load_reference_floors
    manifest=tmp_path/'manifest.json'; manifest.write_text('{"clips":[]}')
    receipt=tmp_path/'contact-floor.json'
    receipt.write_text(json.dumps(dict(schema_version=1,prepared_manifest_sha256='f'*64,clips=[])))
    with pytest.raises(ValueError,match='manifest'):
        load_reference_floors(receipt,manifest,'q1',[],'cpu')
