"""Synthetic rendering fixtures never create a real research holdout."""
from pathlib import Path
import pytest
from scripts.m16_report import render
from spectra_reliability.identity import strict_json, write_json


def fixture(tmp_path):
    out=tmp_path/'synthetic';bench=tmp_path/'synthetic_benchmark';dest=tmp_path/'rendered'
    write_json(out/'confirmation_complete.json',{'synthetic':True})
    write_json(out/'confirmation_freeze.json',{'source_commit':'1'*40,'synthetic':True})
    write_json(out/'independent_replay.json',{'pass':True,'online_answers_checked':8,'fixed_pool_answers_checked':24})
    for name in ('confirmation_opened.json','lineage/inventory.json','confirmation_environment.json'):write_json(out/name,{'synthetic':True})
    latency={'mean_ms':2.,'median_ms':1.,'p95_ms':3.}
    aggregate={'comparisons_to_native_semantic_exit':{'baseline_first_best_first_survival':{
        'twenty_percent_mean_cost_matched_quality_gate':False,'regressions':0,'gains':0,
        'mean_latency_ratio':{'point':1.5,'ci95':[1.1,1.9]}}},
        'fixed_pool':{k:{'selection_failures_given_coverage':v} for k,v in [('improvement',2),('quality',0),('validity',1)]},
        'online':{'synthetic_arm':{'valid_answers':2,'model_example_rows':4,'unique_puzzles':2,'wall_latency':latency}}}
    for stage in ('validation','development','confirmation','shift'):write_json(out/'evaluation'/stage/'aggregate.json',aggregate)
    write_json(bench/'full_graph_summary.json',{'scope':'synthetic_fixture','summaries':{'synthetic_native':{
        'wall':latency,'all_logits_bit_identical':True,'all_answers_equal':True}}})
    write_json(bench/'kernel_summary.json',{'synthetic':True})
    return out,bench,dest


def test_report_renderer_preserves_failed_gate_and_fidelity_boundary(tmp_path):
    out,bench,dest=fixture(tmp_path);render(out,bench,dest)
    summary=strict_json((dest/'summary.json').read_bytes())
    assert summary['primary_controller_twenty_percent_mean_cost_gate'] is False
    assert summary['hiring_or_level_guarantee'] is False
    assert summary['native_scope']=='synthetic_fixture'
    text=(dest/'REPORT.md').read_text()
    assert 'NOT MET' in text and 'not a latency win' in text and 'not evidence of task capability' in text
    assert (dest/'file_hashes.json').is_file() and (dest/'result_table.csv').is_file()


def test_report_renderer_rejects_failed_independent_replay(tmp_path):
    out,bench,dest=fixture(tmp_path);write_json(out/'independent_replay.json',{'pass':False})
    with pytest.raises(ValueError,match='replay'):render(out,bench,dest)
