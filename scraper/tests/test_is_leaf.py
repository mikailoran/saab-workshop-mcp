"""Tests for is_leaf() against the live site.

These hit saabwisonline.com directly rather than cached fixtures on purpose:
a snapshot would keep passing even if the site's markup changed and broke
the real crawl. A failure here means either our detection logic broke, or
the site's page structure changed underneath us -- both worth knowing.
"""
import pytest
import requests
from bs4 import BeautifulSoup

from crawl import USER_AGENT, is_leaf

# An actual procedure/document page: has real content, should be a leaf.
LEAF_URLS = [
    "https://saabwisonline.com/9-3-9440/2007/2-engine/4-cyl-petrol-e85/basic-engine/technical-data/general-data/",
]

# Folder-listing pages: nothing but links, should not be leaves.
NON_LEAF_URLS = [
    "https://saabwisonline.com/9-3-9440/2007/",
    "https://saabwisonline.com/9-3-9440/2007/1-service/saab-service-eu/",
]


def _fetch_content_div(url: str):
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    content_div = soup.find("div", id="content")
    assert content_div is not None, f"site structure changed: no #content div at {url}"
    return content_div


@pytest.mark.parametrize("url", LEAF_URLS)
def test_is_leaf_true_for_document_pages(url):
    assert is_leaf(_fetch_content_div(url)) is True


@pytest.mark.parametrize("url", NON_LEAF_URLS)
def test_is_leaf_false_for_listing_pages(url):
    assert is_leaf(_fetch_content_div(url)) is False