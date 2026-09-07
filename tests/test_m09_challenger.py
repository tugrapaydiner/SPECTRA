from pathlib import Path
from types import SimpleNamespace

import torch

from eval.action_checkpoint import load_m09_action_checkpoint, save_m09_action_checkpoint
from eval.action_search import StateConditionedLatentNativeMCTS
from eval.latent_mcts import _quantize_int8
from model.latent_action import BudgetAlignedChallengerCodebook


class _ToyModel(torch.nn.Module):
    def __init__(self, dim=4):
        super().__init__()
        self.dim = dim; self.T = 1; self.n = 1
        self.token_embed = torch.nn.Embedding(10, dim)
    def encode_positions(self, x, height, width):
        return torch.zeros(1, x.shape[1], self.dim)
    def recursive_cycle(self, x_emb, y, z):
        return y, z


class _ToyMCTS(StateConditionedLatentNativeMCTS):
    def _step(self, x_emb, node, action):
        self.last_search_stats["transition_calls"] = int(self.last_search_stats["transition_calls"]) + 1
        y = node.y.clone(); y[..., 0] = float(action)
        codes, scale = _quantize_int8(node.latent())
        return y, codes, scale
    def _value(self, x, node):
        return 0.05


def _core():
    return SimpleNamespace(
        sha256="core", task={"task":"sudoku"},
        architecture={"dim":4,"num_tokens":10,"seq_len":81}, device=torch.device("cpu")
    )


def _prov():
    return {
        "reference_target_used": False, "candidate_bank_seed": 2901,
        "candidate_bank_count":24, "selected_candidate_indices":[22,15,20],
        "target_utility_table_sha256":"u", "reasoner_tensor_state_sha256":"r",
        "train_id_sha256":"t", "development_id_sha256":"d", "test_id_sha256":"e",
        "target_generation_work":{"state_action_equivalents":24000},
        "training_method":"v3_budget_aligned_challenger",
    }


def test_challenger_prior_is_identity_plus_one_state_challenger():
    m = BudgetAlignedChallengerCodebook(dim=4, num_tokens=10, hidden_dim=8)
    with torch.no_grad():
        for p in m.parameters(): p.zero_()
        m.policy_head.bias[:] = torch.tensor([-2.0, 4.0, -1.0])
    x=torch.zeros(2,4,dtype=torch.long); y=torch.zeros(2,4,4); z=torch.zeros_like(y)
    p=m.priors_for_state(x,y,z)
    assert torch.allclose(p.sum(1), torch.ones(2))
    assert torch.allclose(p[:,0], torch.full((2,),0.5001))
    assert torch.allclose(p[:,2], torch.full((2,),0.4999))
    assert torch.equal((p > 0).sum(1), torch.full((2,),2))


def test_challenger_m08_schedule_is_identity_then_predicted_challenger():
    model=_ToyModel(); m=BudgetAlignedChallengerCodebook(dim=4,num_tokens=10,hidden_dim=8)
    with torch.no_grad():
        for p in m.parameters(): p.zero_()
        m.policy_head.bias[:] = torch.tensor([-2.0,4.0,-1.0])
    s=_ToyMCTS(model,None,m.eval(),2,2,n_rollouts=2,c_puct=1.5,max_depth=1)
    s.search(torch.zeros(1,4,dtype=torch.long))
    assert s.last_search_stats["evaluated_paths"] == [[0],[2]]
    assert s.last_search_stats["transition_calls"] == 4
    assert s.last_search_stats["verifier_evaluations"] == 2


def test_challenger_checkpoint_roundtrip(tmp_path: Path):
    core=_core(); m=BudgetAlignedChallengerCodebook(dim=4,num_tokens=10,hidden_dim=8)
    opt=torch.optim.AdamW(m.parameters(),lr=1e-3)
    # Populate optimizer state and prove this family is restorable.
    x=torch.zeros(2,81,dtype=torch.long); y=torch.randn(2,81,4); z=torch.randn_like(y)
    loss=m.policy_logits_for_state(x,y,z).square().mean()+m.directions.square().mean()
    opt.zero_grad(); loss.backward(); opt.step()
    path=tmp_path/'v3.pt'
    save_m09_action_checkpoint(m,path,core=core,optimizer=opt,trained_steps=1,
                               fitted_version='v3_budget_aligned_challenger',provenance=_prov())
    loaded=load_m09_action_checkpoint(path,core)
    assert isinstance(loaded.module,BudgetAlignedChallengerCodebook)
    assert loaded.payload['architecture']['prior_semantics']=='identity_plus_state_challenger_v1'
