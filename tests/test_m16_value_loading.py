from pathlib import Path
import pytest
import torch
from data.ancestry import digest
from eval.verified_search import ValueTarget
from model.grounded_verifier import GroundedStateVerifier
from model.typed_value import TypedStateValue, load_m16_value

A, B = digest(b'core'), digest(b'train')


def payload(target=ValueTarget.TERMINAL):
    torch.manual_seed(191)
    model = TypedStateValue(target, num_tokens=5, dim=64, n_layers=1, heads=4,
                            max_grid_size=8, act_bits=8, include_y=True)
    return {'format':'spectra.m16_typed_verifier.v1','target':target.value,
            'core_sha256':A, 'architecture':{'dim':64,'n_layers':1,'heads':4,'act_bits':8},
            'model_state':model.state_dict(), 'data_ancestry':{
                'parent_checkpoints':[A],
                'consumed_manifests':[{'sha256':B,'splits':['train'],'role':'training'}]}}


def load(path, **kw):
    return load_m16_value(path,expected_sha256=digest(path.read_bytes()),
                          expected_core_sha256=A,expected_training_manifest_sha256=B,**kw)


def test_typed_wrapper_preserves_initialized_neural_body():
    kwargs = dict(num_tokens=5,dim=64,n_layers=1,heads=4,max_grid_size=8,act_bits=8,include_y=True)
    torch.manual_seed(191); old = GroundedStateVerifier(**kwargs).eval()
    torch.manual_seed(191); new = TypedStateValue(ValueTarget.QUALITY, **kwargs).eval()
    assert old.state_dict().keys() == new.state_dict().keys()
    for k,v in old.state_dict().items(): assert torch.equal(v,new.state_dict()[k])
    x=torch.ones(2,16,dtype=torch.long); y=torch.randn(2,16,64); z=torch.randn_like(y)
    assert torch.equal(old.value_state(x,y,z,4), new.value_state(x,y,z,4))


def test_loader_binds_target_weights_and_independent_core_identity(tmp_path):
    p=tmp_path/'value.pt'; torch.save(payload(),p)
    model,contract=load(p)
    assert model.target is contract.target is ValueTarget.TERMINAL
    assert not model.training and not any(q.requires_grad for q in model.parameters())
    with pytest.raises(ValueError,match='another reasoner'):
        load_m16_value(p,expected_sha256=digest(p.read_bytes()),expected_core_sha256=B,
                       expected_training_manifest_sha256=B)
    with pytest.raises(ValueError,match='hash'):
        load_m16_value(p,expected_sha256=A,expected_core_sha256=A,expected_training_manifest_sha256=B)


@pytest.mark.parametrize('field',['format','architecture','data_ancestry','model_state','unknown'])
def test_malformed_checkpoint_cannot_be_silently_adapted(tmp_path,field):
    obj=payload()
    if field=='model_state': obj[field].pop(next(iter(obj[field])))
    elif field=='unknown': obj[field]=True
    else: obj[field]={}
    p=tmp_path/'value.pt';torch.save(obj,p)
    with pytest.raises((ValueError,RuntimeError)): load(p)


def test_improvement_loading_requires_diagnostic_opt_in(tmp_path):
    p=tmp_path/'value.pt';torch.save(payload(ValueTarget.IMPROVEMENT),p)
    with pytest.raises(ValueError,match='diagnostic'): load(p)
    _, c=load(p,diagnostic_improvement=True)
    with pytest.raises(ValueError,match='improvement'):
        c.bind_selection(model_sha256=A,transition_id='fp64_cycle_action_v1')
