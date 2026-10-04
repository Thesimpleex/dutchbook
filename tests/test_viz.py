import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from dutchbook.viz import THEMES, picture_html, save_figure, sequential_cmap, style


def test_save_figure_renders_light_and_dark(tmp_path):
    seen = []

    def build(palette):
        seen.append(palette["surface"])
        fig, ax = plt.subplots(figsize=(3, 2))
        ax.plot([0, 1], [0, 1])
        return fig

    paths = save_figure(build, tmp_path / "figs" / "demo")
    assert [p.name for p in paths] == ["demo-light.png", "demo-dark.png"]
    assert all(p.stat().st_size > 1000 for p in paths)
    assert seen == [THEMES["light"]["surface"], THEMES["dark"]["surface"]]


def test_style_context_restores_rc_params():
    before = plt.rcParams["figure.facecolor"]
    with style("dark") as palette:
        assert plt.rcParams["figure.facecolor"] == palette["surface"]
    assert plt.rcParams["figure.facecolor"] == before


def test_picture_html_and_cmap():
    html = picture_html("docs/figures/x", "alt text", width=640)
    assert 'srcset="docs/figures/x-dark.png"' in html
    assert 'src="docs/figures/x-light.png"' in html
    assert 'width="640"' in html
    assert sequential_cmap("dark")(0.0) != sequential_cmap("light")(0.0)
