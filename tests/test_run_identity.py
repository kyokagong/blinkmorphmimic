import hashlib
import json
from pathlib import Path
import tempfile
import unittest


class RunIdentityTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.manifest=self.root/'manifest.json'
        self.manifest.write_text(json.dumps({'complete':True,'clips':[]}))
        self.digest=hashlib.sha256(self.manifest.read_bytes()).hexdigest()
        self.source=self.root/'source';(self.source/'src').mkdir(parents=True)
        (self.source/'src'/'model.py').write_text('model = 1\n')
        (self.source/'src'/'private.env').write_text('secret=not included')
        (self.source/'pyproject.toml').write_text('[project]\nname="x"\n')

    def tearDown(self):self.tmp.cleanup()

    def test_resume_requires_exact_manifest_digest(self):
        from morphmimic.train.run_identity import ManifestIdentity
        identity=ManifestIdentity(self.manifest)
        self.assertEqual(identity.sha256,self.digest)
        identity.verify_checkpoint({'optimized_training':{'prepared_manifest_sha256':self.digest}})
        for ckpt in ({},{'optimized_training':{}},{'optimized_training':{'prepared_manifest_sha256':'0'*64}}):
            with self.assertRaises(ValueError):identity.verify_checkpoint(ckpt)

    def test_freeze_receipt_is_no_overwrite_and_rejects_tampering(self):
        from morphmimic.train.run_identity import ManifestIdentity,create_receipt,write_receipt,read_receipt
        identity=ManifestIdentity(self.manifest);path=self.root/'receipt.json'
        receipt=create_receipt(self.source,'sha256:'+'a'*64,{'num_envs':4096},identity.sha256)
        self.assertNotIn('src/private.env',receipt['source_files'])
        write_receipt(path,receipt)
        self.assertEqual(read_receipt(path,identity.sha256),receipt)
        identity.verify_checkpoint({'optimized_training':{'prepared_manifest_sha256':self.digest,'freeze_receipt':receipt}},path)
        with self.assertRaises(FileExistsError):write_receipt(path,receipt)
        receipt['source_files']['src/model.py']='0'*64
        path.write_text(json.dumps(receipt))
        with self.assertRaises(ValueError):read_receipt(path,identity.sha256)

    def test_receipt_rejects_malformed_image_config_and_manifest(self):
        from morphmimic.train.run_identity import create_receipt
        for image,config,digest in [('latest',{},self.digest),('sha256:'+'a'*64,{'x':float('nan')},self.digest),('sha256:'+'a'*64,{},'x')]:
            with self.assertRaises(ValueError):create_receipt(self.source,image,config,digest)

class ReceiptValidationTests(RunIdentityTests):
    def test_bad_receipt_rejected_before_file_creation(self):
        from morphmimic.train.run_identity import write_receipt
        path=self.root/'bad.json'
        with self.assertRaises(ValueError):write_receipt(path,{})
        self.assertFalse(path.exists())

    def test_runtime_source_and_config_mismatch_rejected(self):
        from morphmimic.train.run_identity import create_receipt,write_receipt,read_receipt
        receipt=create_receipt(self.source,'sha256:'+'a'*64,{'num_envs':4096},self.digest)
        path=self.root/'receipt.json';write_receipt(path,receipt)
        read_receipt(path,self.digest,source_root=self.source,image_id='sha256:'+'a'*64,config={'num_envs':4096})
        for kwargs in ({'image_id':'sha256:'+'b'*64},{'config':{'num_envs':32}}):
            with self.assertRaises(ValueError):read_receipt(path,self.digest,**kwargs)
        (self.source/'src'/'model.py').write_text('model = 2\n')
        with self.assertRaises(ValueError):read_receipt(path,self.digest,source_root=self.source)

class FreezeCLITests(RunIdentityTests):
    def test_cli_creates_bound_receipt(self):
        import os,subprocess,sys
        config=self.root/'cfg.json';config.write_text('{"iterations":10000}')
        out=self.root/'freeze.json'
        cmd=[sys.executable,'-m','scripts.freeze_optimized','--data-manifest',str(self.manifest),
             '--source-root',str(self.source),'--config',str(config),'--image-id','sha256:'+'c'*64,'--output',str(out)]
        subprocess.run(cmd,check=True,env=dict(os.environ,PYTHONPATH='src:.'),capture_output=True,text=True)
        from morphmimic.train.run_identity import read_receipt
        self.assertEqual(read_receipt(out,self.digest)['runtime_image_id'],'sha256:'+'c'*64)

class ManifestMutationTests(RunIdentityTests):
    def test_frozen_resume_cannot_drop_receipt_requirement(self):
        from morphmimic.train.run_identity import ManifestIdentity,create_receipt
        receipt=create_receipt(self.source,'sha256:'+'a'*64,{'num_envs':4096},self.digest)
        with self.assertRaisesRegex(ValueError,'requires its freeze receipt'):
            ManifestIdentity(self.manifest).verify_checkpoint({'optimized_training':{'prepared_manifest_sha256':self.digest,'freeze_receipt':receipt}})

    def test_resume_rejects_missing_or_replaced_freeze_receipt(self):
        from morphmimic.train.run_identity import ManifestIdentity,create_receipt,write_receipt
        receipt=create_receipt(self.source,'sha256:'+'a'*64,{'num_envs':4096},self.digest)
        path=self.root/'receipt.json';write_receipt(path,receipt)
        identity=ManifestIdentity(self.manifest)
        for saved in (None,dict(receipt,runtime_image_id='sha256:'+'b'*64)):
            with self.subTest(saved=saved),self.assertRaises(ValueError):
                identity.verify_checkpoint({'optimized_training':{'prepared_manifest_sha256':self.digest,'freeze_receipt':saved}},path)

    def test_changed_manifest_rejected_on_fresh_resume(self):
        from morphmimic.train.run_identity import ManifestIdentity
        old=ManifestIdentity(self.manifest)
        ckpt={'optimized_training':{'prepared_manifest_sha256':old.sha256}}
        self.manifest.write_text('{"complete":true,"clips":["changed"]}')
        with self.assertRaises(ValueError):ManifestIdentity(self.manifest).verify_checkpoint(ckpt)

    def test_invalid_receipt_document_and_data_hash_are_rejected(self):
        from morphmimic.train.run_identity import create_receipt,write_receipt,read_receipt
        receipt=create_receipt(self.source,'sha256:'+'a'*64,{},self.digest)
        path=self.root/'receipt.json';write_receipt(path,receipt)
        with self.assertRaises(ValueError):read_receipt(path,'0'*64)
        path.write_text('{"schema_version":1,"schema_version":1}')
        with self.assertRaises(ValueError):read_receipt(path,self.digest)


class TrainerReceiptTests(RunIdentityTests):
    def test_distributed_device_failure_exits_before_group_creation(self):
        import os
        from unittest.mock import patch
        from morphmimic.scripts.train import run
        from morphmimic.train.trainer import TrainCfg
        with patch.dict(os.environ,{'RANK':'0','LOCAL_RANK':'0'}),patch('torch.cuda.set_device',side_effect=RuntimeError('invalid GPU')),patch('torch.distributed.is_initialized',return_value=False),patch('torch.distributed.init_process_group') as initialize:
            with self.assertRaises(SystemExit) as error:run(TrainCfg(),None,distributed=True)
            self.assertEqual(error.exception.code,1);initialize.assert_not_called()

    def test_distributed_abort_failure_keeps_nonzero_exit_and_original_error(self):
        import io,os
        from contextlib import redirect_stderr
        from unittest.mock import patch
        from morphmimic.scripts.train import run
        from morphmimic.train.trainer import TrainCfg
        errors=io.StringIO()
        with redirect_stderr(errors),patch.dict(os.environ,{'RANK':'0','LOCAL_RANK':'0'}),patch('torch.cuda.set_device'),patch('torch.distributed.init_process_group'),patch('torch.distributed.is_initialized',return_value=True),patch('torch.distributed.distributed_c10d._get_default_group') as group,patch('morphmimic.scripts.train.configure_torch_backends'),patch('morphmimic.scripts.train.Trainer',side_effect=ValueError('original training failure')):
            group.return_value.abort.side_effect=RuntimeError('abort cleanup failure')
            with self.assertRaises(SystemExit) as error:run(TrainCfg(),None,distributed=True)
            self.assertEqual(error.exception.code,1)
        self.assertIn('original training failure',errors.getvalue())
        self.assertIn('abort cleanup failure',errors.getvalue())

    def test_distributed_initialization_error_escapes_without_collective_teardown(self):
        import os
        from unittest.mock import patch
        from morphmimic.scripts.train import run
        from morphmimic.train.trainer import TrainCfg
        with patch.dict(os.environ,{'RANK':'0','LOCAL_RANK':'0'}),patch('torch.cuda.set_device'),patch('torch.distributed.init_process_group'),patch('torch.distributed.is_initialized',return_value=True),patch('torch.distributed.distributed_c10d._get_default_group') as group,patch('torch.distributed.destroy_process_group') as destroy,patch('morphmimic.scripts.train.configure_torch_backends'),patch('morphmimic.scripts.train.Trainer',side_effect=ValueError('unsupported reset state')):
            with self.assertRaises(SystemExit) as error:
                run(TrainCfg(),None,distributed=True)
            self.assertEqual(error.exception.code,1)
            destroy.assert_not_called()
            group.return_value.abort.assert_called_once()

    def test_training_runner_supplies_resume_intent_before_initialization(self):
        from types import SimpleNamespace
        from unittest.mock import Mock,patch
        from morphmimic.scripts.train import run
        from morphmimic.train.trainer import TrainCfg
        trainer=SimpleNamespace(multi=False,is_main=False,train=Mock(),close=Mock(),load=Mock(return_value=[]))
        with patch('morphmimic.scripts.train.Trainer',return_value=trainer) as constructor,patch('morphmimic.scripts.train.configure_torch_backends'):
            for resume in (None,'/fixed/ckpt_0.pt'):
                run(TrainCfg(),resume)
                self.assertEqual(constructor.call_args.kwargs,{'resume':resume is not None})

    def test_trainer_binds_original_launch_config_and_requires_runtime_image(self):
        import os
        from dataclasses import asdict,replace
        from unittest.mock import patch
        from morphmimic.train.run_identity import create_receipt,write_receipt
        from morphmimic.train.trainer import Trainer,TrainCfg
        path=self.root/'receipt.json';image='sha256:'+'a'*64
        cfg=TrainCfg(robot='q1',subset_yaml='/unused.yaml',num_envs=32,
            optimization_schema=1,prepared_manifest=str(self.manifest),freeze_receipt=str(path))
        receipt=create_receipt(Path(__file__).resolve().parents[1],image,asdict(cfg),self.digest)
        write_receipt(path,receipt)
        class EnvReached(RuntimeError):pass
        with patch('morphmimic.train.trainer.MorphMimicEnv',side_effect=EnvReached),patch.dict(os.environ,{'BLINKMORPH_RUNTIME_IMAGE_ID':image}):
            with self.assertRaises(EnvReached):Trainer(cfg,'cpu')
            # Each rank resolves different robot fields; the receipt binds the
            # common launch config supplied before that resolution.
            with patch.object(Trainer,'_resolve_cfg',return_value=replace(cfg,robot='bumi')):
                with self.assertRaises(EnvReached):Trainer(cfg,'cpu')
            for change in ({'compile':'none'},{'wm_batch':16},{'optimizer':'muon'},{'horizon':8}):
                with self.subTest(change=change),self.assertRaisesRegex(ValueError,'Configuration differs'):
                    Trainer(replace(cfg,**change),'cpu')
        with patch('morphmimic.train.trainer.MorphMimicEnv',side_effect=EnvReached),patch.dict(os.environ,{},clear=True):
            with self.assertRaisesRegex(ValueError,'runtime image'):
                Trainer(cfg,'cpu')
