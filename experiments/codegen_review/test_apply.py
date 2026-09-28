"""Patch only recognized source copies; never silently rewrite installed code."""
import importlib.util
import os
from pathlib import Path
import shutil

import pytest

spec=importlib.util.spec_from_file_location('checked_apply',Path(__file__).with_name('apply.py'))
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

@pytest.fixture
def source():
    value=os.environ.get('M2CGEN_ORIGINAL_SOURCE')
    if value is None:pytest.skip('requires an explicit original package source')
    return Path(value)

@pytest.mark.parametrize('variant',['stock','empty','guard','single'])
def test_known_copy_isolated(source,tmp_path,variant):
    before=(source/'m2cgen/ast.py').read_bytes()
    out=tmp_path/'copy';result=module.apply(source,out,variant)
    assert (source/'m2cgen/ast.py').read_bytes()==before==(out/'m2cgen/ast.py').read_bytes()
    assert result['variant']==variant
    with pytest.raises(ValueError):module.apply(source,out,variant)

def test_unknown_ast_rejected_before_write(source,tmp_path):
    changed=tmp_path/'changed';shutil.copytree(source,changed)
    with (changed/'m2cgen/ast.py').open('ab') as f:f.write(b'\n# Changed semantics not validated\n')
    with pytest.raises(ValueError,match='unvalidated'):module.apply(changed,tmp_path/'result','single')
    assert not (tmp_path/'result').exists()

def test_unknown_interpreter_rejected(source,tmp_path):
    changed=tmp_path/'changed';shutil.copytree(source,changed)
    with (changed/'m2cgen/interpreters/interpreter.py').open('ab') as f:f.write(b'\n# Changed interpreter\n')
    with pytest.raises(ValueError,match='unvalidated'):module.apply(changed,tmp_path/'result','single')

def test_nested_destination_rejected(source):
    with pytest.raises(ValueError):module.apply(source,source/'new-package','single')
