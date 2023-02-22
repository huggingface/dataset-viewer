import csv
import urllib.parse
from typing import *



def encode_section_path(page_id, section_path):
    elements = [page_id] + section_path

    return '/'.join([urllib.parse.quote(elem) for elem in elements])
    # return urllib.parse.urlencode({'page':page_id, 'sectionpath':section_path})

def encode_page_only(page_id):
    return urllib.parse.quote(page_id)


class RankingEntry(object):
    """
    A paragraph within a Wikipedia page.

    Attributes:
      paragraph    The content of the Paragraph (which in turn contain a list of ParaBodys)
    """
    def __init__(self, query_id:str, paragraph_id:str, rank:int, score:float, exp_name:str=None, paragraph_content:str=None):
        assert(rank > 0)
        self.query_id = query_id
        self.paragraph_id = paragraph_id
        self.rank = rank
        self.score = score
        self.exp_name = exp_name
        self.paragraph_content = paragraph_content

    def to_trec_eval_row(self, alternative_exp_name=None, page_only=False):
        exp_name_ = alternative_exp_name if alternative_exp_name is not  None \
                    else self.exp_name
        return [self.query_id, 'Q0', self.paragraph_id, self.rank, self.score, exp_name_]

#
# csv.register_dialect(
#     'trec_eval',
#     delimiter = ' ',
#     quotechar = '"',
#     doublequote = False,
#     skipinitialspace = False,
#     lineterminator = '\n',
#     quoting = csv.QUOTE_NONE)
#
#
# def configure_csv_writer(fileobj):
#     'Convenience method to create a csv writer with the trec_eval_dialect'
#     return csv.writer(fileobj, dialect='trec_eval')
#

def format_run(writer, ranking_of_paragraphs, exp_name=None):
    'write one ranking to the csv writer'
    for elem in ranking_of_paragraphs:
        # query-number    Q0  document-id rank    score   Exp
        writer.write(" ".join([str(x) for x in elem.to_trec_eval_row(exp_name)]))
        writer.write("\n")
