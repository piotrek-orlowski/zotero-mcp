"""Tests for zotero_mcp.text_filters.strip_boilerplate.

Each positive fixture is a concatenation of a publisher boilerplate
block followed by an ``Abstract`` heading and a sentinel content line.
Every positive assertion therefore proves both that the boilerplate is
removed AND that legitimate content past the boilerplate survives.
"""
from __future__ import annotations

import pytest

from zotero_mcp.text_filters import strip_boilerplate


_TAIL = "\nAbstract\nThis is the paper's actual content.\n"


FIX_ECONSTOR = (
    "A Service of\n"
    "zbw\n"
    "Leibniz-Informationszentrum\n"
    "Wirtschaft\n"
    "Make Your Publications Visible.\n"
    "\n"
    "Smith, John\n"
    "Working Paper\n"
    "The economics of something important\n"
    "\n"
    "CFR Working Paper, No. 21-08\n"
    "\n"
    "Provided in Cooperation with:\n"
    "Centre for Financial Research (CFR), University of Cologne\n"
    "\n"
    "Suggested Citation: Smith, John (2021) : The economics of something important,\n"
    "CFR Working Paper, No. 21-08, University of Cologne, Centre for Financial Research (CFR),\n"
    "Cologne\n"
    "\n"
    "This Version is available at:\n"
    "https://hdl.handle.net/10419/123456\n"
    "\n"
    "Standard-Nutzungsbedingungen:\n"
    "Die Dokumente auf EconStor dürfen zu eigenen wissenschaftlichen Zwecken\n"
    "und zum Privatgebrauch gespeichert und kopiert werden.\n"
    "\n"
    "Terms of use:\n"
    "Documents in EconStor may be saved and copied for your personal and\n"
    "scholarly purposes.\n"
) + _TAIL

FIX_SSRN = (
    "SSRN Electronic Journal\n"
    "\n"
    "Electronic copy available at: https://ssrn.com/abstract=1234567\n"
    "\n"
    "Electronic copy available at: https://ssrn.com/abstract=1234567\n"
) + _TAIL

FIX_JSTOR = (
    "Some Paper Title\n"
    "Author: A. N. Author\n"
    "Source: The Journal Name, Vol. 10, No. 2\n"
    "Published by: Some Publisher\n"
    "Stable URL: https://www.jstor.org/stable/1234567\n"
    "Accessed: 01-01-2020 00:00 UTC\n"
    "\n"
    "Your use of the JSTOR archive indicates your acceptance of the Terms & Conditions of Use.\n"
    "\n"
    "JSTOR is a not-for-profit service that helps scholars, researchers, and students discover, use, and\n"
    "build upon a wide range of content in a trusted digital archive. We use information technology and\n"
    "tools to increase productivity and facilitate new forms of scholarship. For more information about\n"
    "JSTOR, please contact support@jstor.org.\n"
    "\n"
    "Some Publisher are collaborating with JSTOR to digitize, preserve and extend access.\n"
    "\n"
    "This content downloaded from 192.168.1.1 on Wed, 01 Jan 2020 00:00:00 UTC\n"
    "All use subject to https://about.jstor.org/terms\n"
) + _TAIL

FIX_ELSEVIER = (
    "Journal of Something 12 (2019) 1-20\n"
    "\n"
    "Contents lists available at ScienceDirect\n"
    "\n"
    "Journal of Something\n"
    "\n"
    "journal homepage: www.elsevier.com/locate/jsomething\n"
    "\n"
    "Article Title\n"
    "\n"
    "A. Author, B. Author\n"
    "\n"
    "a r t i c l e i n f o\n"
    "\n"
    "Article history:\n"
    "Received 1 Jan 2019\n"
    "\n"
    "1234-5678/© 2019 Elsevier B.V. All rights reserved.\n"
) + _TAIL

FIX_OUP = (
    "The Review of Financial Studies / v 31 n 10 2018\n"
    "\n"
    "For Permissions, please e-mail: journals.permissions@oup.com\n"
) + _TAIL

FIX_INFORMS = (
    "This article was downloaded by: [some ip]\n"
    "On: 01 Jan 2020, At: 00:00\n"
    "Publisher: Institute for Operations Research and the Management Sciences (INFORMS)\n"
    "INFORMS is located in Maryland, USA\n"
    "\n"
    "Management Science\n"
    "\n"
    "Publication details, including instructions for authors and subscription information:\n"
    "http://pubsonline.informs.org\n"
    "\n"
    "Please scroll down for article—it is on subsequent pages\n"
    "\n"
    "With 12,500 members from nearly 90 countries, INFORMS is the largest international\n"
    "association of operations research (O.R.) and analytics professionals and students.\n"
    "INFORMS provides unique networking and learning opportunities for individual\n"
    "professionals, and organizations of all types and sizes, to better understand and use\n"
    "O.R. and analytics tools and methods to transform strategic visions and achieve better outcomes.\n"
    "\n"
    "ISSN 0025-1909 (print), ISSN 1526-5501 (online)\n"
    "\n"
    "Full terms and conditions of use: https://pubsonline.informs.org/page/terms-and-conditions\n"
    "\n"
    "Copyright © 2020, INFORMS\n"
) + _TAIL

FIX_TANDF = (
    "Quantitative Finance\n"
    "\n"
    "ISSN: 1469-7688 (Print) 1469-7696 (Online) Journal homepage\n"
    "\n"
    "Article Title Here\n"
    "\n"
    "To link to this article: https://doi.org/10.1080/12345678.2020.1234567\n"
    "\n"
    "Submit your article to this journal\n"
    "\n"
    "View related articles\n"
    "\n"
    "View Crossmark data\n"
    "\n"
    "Full Terms & Conditions of access and use can be found at\n"
    "\n"
    "This article may be used only for the purposes of research, teaching, and private study.\n"
) + _TAIL

FIX_WILEY = (
    "Some Wiley Title\n"
    "Author(s): A. Author and B. Author\n"
    "Source: The Journal of Finance, Vol. 70, No. 5 (October 2015)\n"
    "Published by: Wiley for the American Finance Association\n"
    "Stable URL: https://www.jstor.org/stable/26654321\n"
    "\n"
    "The Journal of Finance\n"
    "\n"
    "© 2015 the American Finance Association\n"
) + _TAIL

FIX_ECONOMETRIC_SOC = (
    "Econometrica, Vol. 80, No. 3 (May, 2012), 1001-1044\n"
    "\n"
    "Published by: The Econometric Society\n"
    "Stable URL: https://www.jstor.org/stable/41408720\n"
    "\n"
    "© 2012 The Econometric Society\n"
) + _TAIL

FIX_JFQA = (
    "JOURNAL OF FINANCIAL AND QUANTITATIVE ANALYSIS\n"
    "\n"
    "MICHAEL G. FOSTER SCHOOL OF BUSINESS\n"
    "\n"
    "doi:10.1017/S0022109012000012\n"
    "\n"
    "https://doi.org/10.1017/S0022109012000012\n"
) + _TAIL

FIX_NBER = (
    "NBER WORKING PAPER SERIES\n"
    "\n"
    "A Paper Title\n"
    "\n"
    "Author Name\n"
    "\n"
    "Working Paper 12345\n"
    "http://www.nber.org/papers/w12345\n"
    "\n"
    "NATIONAL BUREAU OF ECONOMIC RESEARCH\n"
    "1050 Massachusetts Avenue\n"
    "Cambridge, MA 02138\n"
    "\n"
    "NBER Working Paper No. 12345\n"
) + _TAIL


# ---------------------------------------------------------------------------
# Positive — one test per publisher block
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "fixture,gone",
    [
        (FIX_ECONSTOR, "Suggested Citation"),
        (FIX_SSRN, "Electronic copy available at"),
        (FIX_JSTOR, "All use subject to https://about.jstor.org/terms"),
        (FIX_ELSEVIER, "Contents lists available at ScienceDirect"),
        (FIX_OUP, "journals.permissions@oup.com"),
        (FIX_INFORMS, "INFORMS is located in Maryland"),
        (FIX_TANDF, "View Crossmark data"),
        (FIX_WILEY, "Wiley for the American Finance Association"),
        (FIX_ECONOMETRIC_SOC, "The Econometric Society"),
        (FIX_JFQA, "MICHAEL G. FOSTER SCHOOL OF BUSINESS"),
        (FIX_NBER, "NATIONAL BUREAU OF ECONOMIC RESEARCH"),
    ],
)
def test_publisher_block_stripped(fixture, gone):
    out = strip_boilerplate(fixture)
    assert gone not in out
    assert "paper's actual content" in out


# ---------------------------------------------------------------------------
# Negative — false-friend preservation
# ---------------------------------------------------------------------------

def test_inline_terms_of_use_survives():
    text = "The terms of use of the contract bound both parties equally.\n"
    assert "terms of use of the contract" in strip_boilerplate(text)


@pytest.mark.parametrize(
    "heading",
    ["1. Introduction", "Introduction", "1 Introduction", "I. Introduction"],
)
def test_section_heading_introduction_survives(heading):
    text = f"{heading}\nThis section introduces the topic.\n"
    assert heading in strip_boilerplate(text)


@pytest.mark.parametrize("heading", ["References", "REFERENCES"])
def test_references_heading_survives(heading):
    text = f"{heading}\nSmith, J. (2020) On things, Journal of Things.\n"
    assert heading in strip_boilerplate(text)


def test_jel_and_keywords_line_survives():
    text = "JEL Classification: G11; G12\nKeywords: options, volatility\n"
    out = strip_boilerplate(text)
    assert "JEL Classification: G11; G12" in out
    assert "Keywords: options, volatility" in out


def test_reference_list_entry_survives():
    text = "Jiang, G., and Y. Tian, 2005, Review of Financial Studies 18, 1305.\n"
    assert "Review of Financial Studies 18, 1305" in strip_boilerplate(text)


# ---------------------------------------------------------------------------
# Ligature / skip-to-abstract / edge cases
# ---------------------------------------------------------------------------

def test_ligatures_normalized():
    # jo + U+FB01 (fi ligature); C + U+20DD (combining enclosing circle)
    text = "jo\ufb01 C\u20dd 2020"
    assert strip_boilerplate(text) == "jofi © 2020"


def test_skip_to_abstract_drops_preamble():
    # Preamble short, body long, so the Abstract heading lies well within
    # the first 33% of the text where _skip_to_abstract looks.
    preamble = "Preamble stuff to drop.\n"
    body = "Body line that repeats.\n" * 20
    text = preamble + "Abstract\n" + body
    out = strip_boilerplate(text)
    assert out.startswith("Abstract")
    assert "Preamble stuff to drop" not in out


def test_skip_to_abstract_noop_when_absent():
    text = "First line of body.\nSecond line of body.\nThird line of body.\n"
    assert strip_boilerplate(text) == text


def test_empty_input_returns_empty():
    assert strip_boilerplate("") == ""


def test_strip_boilerplate_disabled_preserves_input():
    # skip_to_abstract=False AND no regex match → input passes through
    # untouched (modulo ligature normalization, which is a no-op here).
    text = "Arbitrary text without any publisher markers.\nAnother line.\n"
    assert strip_boilerplate(text, skip_to_abstract=False) == text
