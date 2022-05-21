import time
import logging
import urllib.request
import feedparser
from typing import List, Dict, Tuple
import requests

from .arxiv_entry import ArxivEntry

logger = logging.getLogger(__name__)


query = 'cat:cs.CV+OR+cat:cs.LG+OR+cat:cs.CL+OR+cat:cs.AI+OR+cat:cs.NE+OR+cat:cs.RO'


"""
Utils for dealing with arxiv API and related processing
"""


def scrape_recent_papers(total_results=5, max_results_per_page = 500, verbose=True) -> List[ArxivEntry]:

    results = [] # type: List[ArxivEntry]
    start_index = 0
    while start_index < total_results:
        if verbose:
            print(f'Querying arxiv for {start_index} to {start_index + max_results_per_page}')
        result_count = min(total_results - start_index, max_results_per_page)
        response = get_response(query, results_per_page=result_count, start_index=start_index)
        batch = parse_response(response)  # type: List[ArxivEntry]
        start_index += len(batch)
        results.extend(batch)
        time.sleep(3)
    return results


def get_response(search_query, start_index=0, results_per_page=100):
    """ hits arxiv.org API to fetch a batch of 100 papers """
    # fetch raw response
    base_url = 'http://export.arxiv.org/api/query?'
    add_url = f'search_query=%s&sortBy=lastUpdatedDate&start=%d&max_results={results_per_page}' % (search_query, start_index)
    #add_url = 'search_query=%s&sortBy=submittedDate&start=%d&max_results=100' % (search_query, start_index)
    search_query = base_url + add_url
    logger.debug(f"Searching arxiv for {search_query}")
    with urllib.request.urlopen(search_query) as url:
        response = url.read()

    if url.status != 200:
        logger.error(f"arxiv did not return status 200 response")

    return response

def encode_feedparser_dict(d):
    """ helper function to strip feedparser objects using a deep copy """
    if isinstance(d, feedparser.FeedParserDict) or isinstance(d, dict):
        return {k: encode_feedparser_dict(d[k]) for k in d.keys()}
    elif isinstance(d, list):
        return [encode_feedparser_dict(k) for k in d]
    else:
        return d

def parse_arxiv_url(url):
    """
    examples is http://arxiv.org/abs/1512.08756v2
    we want to extract the raw id (1512.08756) and the version (2)
    """
    ix = url.rfind('/')
    assert ix >= 0, 'bad url: ' + url
    idv = url[ix+1:] # extract just the id (and the version)
    parts = idv.split('v')
    assert len(parts) == 2, 'error splitting id and version in idv string: ' + idv
    return idv, parts[0], int(parts[1])

def cleanup_title(title: str) -> str:
    """
    remove spaces and other non-alphanumeric characters from the title
    """
    return ' '.join(title.split())

def parse_published_date_to_unix_ts(published_date: str) -> int:
    """
    convert arxiv published date to unix timestamp
    """
    return int(time.mktime(time.strptime(published_date, '%Y-%m-%dT%H:%M:%SZ')))

def clean_abstract(abstract: str) -> str:
    """
    remove newlines and other whitespace characters from the abstract
    """
    return ' '.join(abstract.split())

def parse_authors(authors: List[Dict[str, str]]) -> List[str]:
    """
    parse the authors from a list of dicts to a list of strings
    """
    return [author['name'] for author in authors]

# Input is response bytes. For libraries input is created via:
#     Requests: response.content
#     Urllib: response.read()
def parse_response(response) -> List[ArxivEntry]:
    out = []
    parse = feedparser.parse(response)
    for e in parse.entries:
        j = encode_feedparser_dict(e)
        # extract / parse id information
        idv, rawid, version = parse_arxiv_url(j['id'])
        comment = None
        if 'arxiv_comment' in j:
            comment = j['arxiv_comment']

        entry = ArxivEntry(
            paper_id=rawid,
            title=cleanup_title(j['title']),
            authors=parse_authors(j['authors']),
            abstract=clean_abstract(j['summary']),
            first_published=parse_published_date_to_unix_ts(j['published']),
            comment=comment,
        )
        out.append(entry)

    return out


def get_paper_info_by_id(paper_id: str) -> List[ArxivEntry]:
    url = "http://export.arxiv.org/api/query?"
    get_url = f'{url}id_list={paper_id}'
    response = requests.get(get_url)
    return parse_response(response.content)
