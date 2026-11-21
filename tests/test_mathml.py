"""The 6 required MathML -> LaTeX cases from PROJECT_SPEC.md §6.4.

Note: msup/msub/msubsup always wrap their base in braces ("{a}^{b}"), per the §6.4
mapping table. This makes "t^{2}" render as "{t}^{2}" — visually identical LaTeX, just
with a redundant (harmless) brace pair — which is why the sqrt test case below, which
exercises the same msup rule twice, matches the spec's literal expected string exactly.
See README.md "Decisions" for the rationale.
"""
from bs4 import BeautifulSoup

from app.content.mathml import mathml_to_latex


def _convert(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    return mathml_to_latex(soup.find("math"))


def _norm(s: str) -> str:
    return " ".join(s.split())


def test_suvat_equation():
    html = (
        "<math><mi>s</mi><mo>=</mo><mi>u</mi><mi>t</mi><mo>+</mo>"
        "<mfrac><mn>1</mn><mn>2</mn></mfrac><mi>a</mi>"
        "<msup><mi>t</mi><mn>2</mn></msup></math>"
    )
    assert _norm(_convert(html)) == _norm("s=ut+\\frac{1}{2}a{t}^{2}")


def test_h2o_subscript():
    html = "<math><msub><mi>H</mi><mn>2</mn></msub><mi>O</mi></math>"
    assert _norm(_convert(html)) == _norm("{H}_{2}O")


def test_msqrt_of_velocity_difference():
    html = (
        "<math><msqrt><msup><mi>v</mi><mn>2</mn></msup><mo>&#8722;</mo>"
        "<msup><mi>u</mi><mn>2</mn></msup></msqrt></math>"
    )
    assert _norm(_convert(html)) == _norm("\\sqrt{{v}^{2}-{u}^{2}}")


def test_mover_vector_force():
    html = "<math><mover><mi>F</mi><mo>&#8594;</mo></mover></math>"
    assert _norm(_convert(html)) == _norm("\\vec{F}")


def test_mfenced_pair():
    html = "<math><mfenced><mi>a</mi><mi>b</mi></mfenced></math>"
    assert _norm(_convert(html)) == _norm("\\left(a,b\\right)")


def test_greek_lambda():
    html = "<math><mi>&#955;</mi></math>"
    assert _norm(_convert(html)) == _norm("\\lambda")
