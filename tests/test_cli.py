import csv
import json
from html.parser import HTMLParser
from pathlib import Path

import pytest

from dutchbook.cli import main
from dutchbook.spec import load_spec, parse_probability, read_forecasts

SPEC = """\
title = "Linda"

[atoms]
F = "Linda is a feminist"
B = "Linda is a bank teller"

[formulas]
bank_teller = "B"
feminist_bank_teller = "F & B"
"""

CSV = """\
group,id,forecast
fallacy,bank_teller,0.15
fallacy,feminist_bank_teller,0.35
sound,bank_teller,0.15
sound,feminist_bank_teller,0.10
"""


@pytest.fixture
def files(tmp_path):
    spec, data = tmp_path / "spec.toml", tmp_path / "f.csv"
    spec.write_text(SPEC)
    data.write_text(CSV)
    return spec, data


def run(capsys, *argv):
    code = main([str(a) for a in argv])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_spec_and_forecast_parsing(files):
    spec_path, data_path = files
    spec = load_spec(spec_path)
    assert spec.ids == ("bank_teller", "feminist_bank_teller")
    assert str(spec.formula("feminist_bank_teller")) == "F ∧ B"
    assert spec.atoms["F"] == "Linda is a feminist"
    sets = read_forecasts(data_path, spec)
    assert [s.group for s in sets] == ["fallacy", "sound"]
    assert sets[0].values.tolist() == [0.15, 0.35]
    with pytest.raises(KeyError):
        spec.formula("nope")


def test_parse_probability_formats():
    assert parse_probability("0.3") == 0.3
    assert parse_probability(" 30% ") == pytest.approx(0.3)
    assert parse_probability(".5") == 0.5
    for bad in ("1.2", "-0.1", "abc", "nan", "150%"):
        with pytest.raises(ValueError, match=r"outside|could not convert"):
            parse_probability(bad)


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("[formulas]\n", "empty"),
        ("[formulas]\nx = 3\n", "must be a string"),
        ('[formulas]\nx = "A &"\n', "formula 'x'"),
        ("not toml [", "invalid TOML"),
        ("title = 'x'\n", "formulas"),
    ],
)
def test_bad_specs(tmp_path, content, message):
    path = tmp_path / "s.toml"
    path.write_text(content)
    with pytest.raises(ValueError, match=message):
        load_spec(path)


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("id,value\nx,1\n", "needs the columns"),
        ("id,forecast\nunknown,0.5\n", "not defined"),
        ("id,forecast\nbank_teller,0.5\nbank_teller,0.6\n", "duplicate"),
        ("id,forecast\nbank_teller,2\n", "outside"),
        ("id,forecast\n", "no forecasts"),
    ],
)
def test_bad_forecast_tables(files, tmp_path, content, message):
    spec = load_spec(files[0])
    path = tmp_path / "bad.csv"
    path.write_text(content)
    with pytest.raises(ValueError, match=message):
        read_forecasts(path, spec)


def test_audit_incoherent_prints_the_dutch_book_and_exits_one(files, capsys):
    spec, data = files
    code, out, _ = run(capsys, "audit", data, "--spec", spec, "--group", "fallacy", "--width", 100)
    assert code == 1
    assert "INCOHERENT" in out
    assert "guarantees 0.100" in out
    assert "BUY" in out
    assert "SELL" in out
    assert "F ∧ B" in out
    assert "Linda is a bank teller" in out
    assert "Guaranteed in every world" in out


def test_audit_coherent_prints_a_witness_and_exits_zero(files, capsys):
    spec, data = files
    code, out, _ = run(capsys, "audit", data, "--spec", spec, "--group", "sound", "--width", 100)
    assert code == 0
    assert "COHERENT" in out
    assert "probability measure" in out


def test_audit_multiple_groups_shows_leaderboard(files, capsys):
    spec, data = files
    code, out, _ = run(capsys, "audit", data, "--spec", spec, "--width", 100)
    assert code == 1
    assert "fallacy" in out
    assert "sound" in out
    assert out.index("fallacy") < out.index("sound")  # worst first


def test_audit_json_html_and_stake_scaling(files, tmp_path, capsys):
    spec, data = files
    js, page = tmp_path / "out" / "a.json", tmp_path / "out" / "a.html"
    code, _, _ = run(
        capsys,
        "audit",
        data,
        "--spec",
        spec,
        "--stake",
        100,
        "--json",
        js,
        "--html",
        page,
        "--width",
        100,
    )
    assert code == 1
    payload = json.loads(js.read_text())
    group = {g["group"]: g for g in payload["groups"]}["fallacy"]
    assert group["guaranteed_profit"] == pytest.approx(10.0)
    assert not group["coherent"]
    assert {o["side"] for o in group["orders"]} == {"BUY", "SELL"}
    assert all(w["net_payoff"] >= 10.0 - 1e-6 for w in group["payoffs"])
    html = page.read_text()
    assert html.startswith("<!doctype html>")
    assert "fallacy" in html
    assert "Coherence leaderboard" in html

    class Check(HTMLParser):
        def __init__(self):
            super().__init__()
            self.stack, self.errors = [], 0

        def handle_starttag(self, tag, attrs):
            if tag not in {"meta", "br", "img"}:
                self.stack.append(tag)

        def handle_endtag(self, tag):
            if tag in {"meta", "br", "img"}:
                return
            if not self.stack or self.stack.pop() != tag:
                self.errors += 1

    parser = Check()
    parser.feed(html)
    assert parser.errors == 0
    assert parser.stack == []


@pytest.mark.parametrize("norm", ["l1", "linf", "l2"])
def test_audit_stake_norms(files, capsys, norm):
    spec, data = files
    expected = {"l1": "0.100", "linf": "0.200", "l2": "0.141"}[norm]
    code, out, _ = run(
        capsys,
        "audit",
        data,
        "--spec",
        spec,
        "--group",
        "fallacy",
        "--stake-norm",
        norm,
        "--width",
        100,
    )
    assert code == 1
    assert f"guarantees {expected}" in out


def test_repair_writes_a_coherent_file_and_certificate(files, tmp_path, capsys):
    spec, data = files
    out_csv, cert = tmp_path / "r.csv", tmp_path / "c.json"
    code, out, _ = run(
        capsys,
        "repair",
        data,
        "--spec",
        spec,
        "--out",
        out_csv,
        "--certificate",
        cert,
        "--width",
        100,
    )
    assert code == 0
    assert "DOMINATED" in out
    rows = list(csv.DictReader(out_csv.open()))
    assert {r["group"] for r in rows} == {"fallacy", "sound"}
    # the repaired file passes the audit
    code, _, _ = run(capsys, "audit", out_csv, "--spec", spec)
    assert code == 0
    certificate = {c["group"]: c for c in json.loads(cert.read_text())}
    assert certificate["fallacy"]["holds"]
    assert (
        certificate["fallacy"]["min_improvement"]
        >= certificate["fallacy"]["guaranteed_improvement"] - 1e-9
    )
    assert certificate["sound"]["guaranteed_improvement"] == pytest.approx(0.0, abs=1e-12)


def test_repair_kl(files, capsys):
    spec, data = files
    code, out, _ = run(
        capsys,
        "repair",
        data,
        "--spec",
        spec,
        "--group",
        "fallacy",
        "--divergence",
        "kl",
        "--width",
        100,
    )
    assert code == 0
    assert "log loss" in out


def test_bounds_command(files, tmp_path, capsys):
    spec, data = files
    code, out, _ = run(
        capsys, "bounds", data, "--spec", spec, "--group", "sound", "--formula", "F", "--width", 100
    )
    assert code == 0
    assert "[0.1000, 0.9500]" in out
    code, out, _ = run(
        capsys, "bounds", data, "--spec", spec, "--group", "fallacy", "--formula", "F"
    )
    assert code == 1
    assert "INCOHERENT" in out
    code, _, err = run(capsys, "bounds", data, "--spec", spec, "--formula", "F")
    assert code == 2
    assert "--group" in err


def test_errors_exit_with_code_two(files, capsys, tmp_path):
    spec, data = files
    code, _, err = run(capsys, "audit", data, "--spec", spec, "--group", "missing")
    assert code == 2
    assert "no forecast set named" in err
    code, _, err = run(capsys, "audit", tmp_path / "nope.csv", "--spec", spec)
    assert code == 2
    assert "dutchbook: error" in err


def test_version_and_help(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert "dutchbook" in capsys.readouterr().out


def test_forecastbench_command_on_a_local_snapshot(snapshot_tar, tmp_path, capsys):
    raw = snapshot_tar.parent
    page = tmp_path / "fb.html"
    argv = [
        "forecastbench",
        "--round",
        "2024-07-21",
        "--raw-dir",
        raw,
        "--cache-dir",
        tmp_path / "c",
        "--no-download",
        "--workers",
        "1",
        "--min-tuples",
        "1",
        "--html",
        page,
        "--width",
        140,
    ]
    code, out, _ = run(capsys, *argv)
    assert code == 0
    assert "round 2024-07-21" in out
    assert "elicitation artefact" in out
    assert "Model-A (zero shot)" in out
    assert "Model-B (zero shot)" in out
    assert "Always 0.5" not in out
    html = page.read_text()
    assert "Per-model coherence leaderboard" in html
    assert "separate prompt" in html
    code, _, err = run(
        capsys,
        "forecastbench",
        "--round",
        "2025-10-26",
        "--raw-dir",
        raw,
        "--cache-dir",
        tmp_path / "c",
        "--no-download",
    )
    assert code == 2
    assert "no combination questions" in err
    code, _, err = run(capsys, "forecastbench", "--raw-dir", tmp_path / "empty", "--no-download")
    assert code == 2
    assert "--no-download" in err


def test_example_files_are_valid():
    root = Path(__file__).resolve().parent.parent / "examples"
    spec = load_spec(root / "linda.toml")
    sets = read_forecasts(root / "linda.csv", spec)
    assert len(sets) == 2


def test_column_generation_method_on_the_command_line(files, capsys):
    spec, data = files
    code, out, _ = run(
        capsys,
        "audit",
        data,
        "--spec",
        spec,
        "--group",
        "fallacy",
        "--method",
        "column-generation",
        "--width",
        100,
    )
    assert code == 1
    assert "guarantees 0.100" in out
    assert "generated worlds" in out
    code, out, _ = run(
        capsys,
        "repair",
        data,
        "--spec",
        spec,
        "--group",
        "fallacy",
        "--method",
        "column-generation",
        "--width",
        100,
    )
    assert code == 0
    assert "DOMINATED" in out
