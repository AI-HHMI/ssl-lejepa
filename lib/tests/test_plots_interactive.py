"""Tests for plots_interactive.py: fragments carry valid Plotly JSON, sortable tables and layered image cells."""

import json
import re
import shutil
import subprocess

import pytest

import plots_interactive as pi
import report_page


def islands(fragment: str) -> list[dict]:
    return [json.loads(s) for s in re.findall(r"<script type='application/json' id='iplot\d+-data'>(.*?)</script>", fragment)]


def test_scatter_has_tooltips_baselines_and_axis_buttons():
    frag = pi.scatter({"linear": [{"EFLOP": 1.0, "AP": 0.5, "run": "a</script>"}]}, "EFLOP", "AP", hlines={"random": 0.1}, logx=True)
    (fig,) = islands(frag)
    assert fig["data"][0]["text"] == ["EFLOP: 1<br>AP: 0.5<br>run: a</script>"]  # escaped in the page, intact once parsed
    assert fig["layout"]["shapes"][0]["name"] == "random" and fig["layout"]["shapes"][0]["y0"] == 0.1
    assert fig["layout"]["xaxis"]["type"] == "log" and fig["layout"]["updatemenus"][0]["active"] == 1


def test_bar_keeps_the_given_order_top_to_bottom():
    (fig,) = islands(pi.bar([("b", 2.0, "g1"), ("a", 1.0, "g2"), ("c", 3.0, "g1")], "x"))
    assert fig["layout"]["yaxis"]["categoryarray"] == ["b", "a", "c"] and [t["name"] for t in fig["data"]] == ["g1", "g2"]
    assert [m["buttons"][0]["label"] for m in fig["layout"]["updatemenus"]] == ["x linear"]  # no log buttons on the labels


def test_table_is_sortable_with_numeric_cells():
    frag = pi.table(["run", "AP"], [["d0", 0.12345], ["d1", "—"]])
    assert "class='sortable'" in frag and "<td class='n'>0.1235</td>" in frag and "<tbody>" in frag


def test_images_embed_every_layer_and_mark_missing_cells(tmp_path):
    em, pca = tmp_path / "em.png", tmp_path / "pca.png"
    for p in (em, pca):
        p.write_bytes(b"\x89PNG\r\n\x1a\n")
    frag = pi.images({"xs": {"c1": [em, pca]}, "s": {"c1": []}})
    assert frag.count("data:image/png;base64,") == 2 and "class='zoom'" in frag and "hidden=''" in frag and "alt='pca'" in frag
    assert "<td>–</td>" in frag and "2 layers" in frag
    with pytest.raises(AssertionError, match="no image"):
        pi.images({"xs": {"c1": [tmp_path / "missing.png"]}})


def test_runtime_only_on_interactive_pages(tmp_path):
    report_page.write(tmp_path / "static.html", "t", "", [], [("S", [("c", "<svg></svg>")])])
    report_page.write(tmp_path / "live.html", "t", "", [], [("S", [("c", pi.table(["a"], [[1]]))])])
    assert "plotly" not in (tmp_path / "static.html").read_text() and "plotly" in (tmp_path / "live.html").read_text()


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node for a JS syntax check")
def test_runtime_js_parses(tmp_path):
    (tmp_path / "runtime.js").write_text(pi.JS)
    subprocess.run(["node", "--check", str(tmp_path / "runtime.js")], check=True)
