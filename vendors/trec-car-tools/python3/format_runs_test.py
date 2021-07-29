from trec_car.format_runs import *
from trec_car.read_data import *
import itertools
import sys

if len(sys.argv)<3:
    print("usage ",sys.argv[0]," outlinefile paragraphfile out")
    exit()

query_cbor=sys.argv[1]
psg_cbor=sys.argv[2]
out=sys.argv[3]

pages = []
with open(query_cbor, 'rb') as f:
    pages = [p for p in itertools.islice(iter_annotations(f), 0, 1000)]


paragraphs = []
with open(psg_cbor, 'rb') as f:
    d = {p.para_id: p for p in itertools.islice(iter_paragraphs(f), 0, 500 ,5)}
    paragraphs = d.values()

print("pages: ", len(pages))
print("paragraphs: ", len(paragraphs))

mock_ranking = [(p, 1.0 / (r + 1), (r + 1)) for p, r in zip(paragraphs, range(0, 1000))]

with open(out,mode='w', encoding='UTF-8') as f:
    writer = f
    numqueries = 0
    for page in pages:
        for section_path in page.flat_headings_list():
            numqueries += 1
            query_id = "/".join([page.page_id]+[section.headingId for section in section_path])
            ranking = [RankingEntry(query_id, p.para_id, r, s, paragraph_content=p) for p, s, r in mock_ranking]
            format_run(writer, ranking, exp_name='test')

    f.close()
    print("num queries = ", numqueries)
