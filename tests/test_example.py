import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from example_demo import query


def test_authored_example_runs_actual_retrieval_core():
    result=query("automatic","open hands",{"threshold":["0.90"]})
    assert result["slots"] and result["rule_count"]>=1
    assert len(result["slots"][0]["frames"])==45
    assert "no trained weights" in result["data_label"]
