# This is an example on how to access content of TREC CAR data
# and convert it into a string of content with offset-based entity link annotations.
# Feel free to use the AnnotatedContentBuilder
# I highly recommend that you implement your own version `annotate_section_content`
# because you need to make decisions on which content to include, where to
# futher provide newlines etc.
# Keep in mind, whatever you add to your output needs to go through the
# AnnotatedContenBuilder or offsets won't match
# you can add all kinds of semantic annotations on offsets. However, in the current
# implementation they much be non-overlapping.



from trec_car.read_data import *

class Annotation():
    """Wraps a semantic annotation with offset information """
    def __init__(self, start, end, annotation):
        self.start = start
        self.end = end
        self.annotation = annotation

class AnnotatedContentBuilder():
    """Builds a string iteratively and keeps track of offsets.
       Chunks of plain text and semantic annotations need to added in order
    """
    def __init__(self):
        self.content = ""
        self.offset = 0
        self.annotations = []

    def append(self, chunk, optAnnotation=None):
        start = self.offset
        self.content += chunk
        self.offset = len(self.content)
        end = self.offset
        if optAnnotation:
            self.annotations.append( Annotation(start=start, end=end, annotation=optAnnotation))

    def get_content(self):
        return self.content

    def get_annotations(self):
        return self.annotations


def annotate_section_content(section):
    """ Example implementation to break out the content of a (top-level) section with entity links """
    def annotated_content(skel, contentBuilder):
            if isinstance(skel, Section):
                contentBuilder.append('\n')
                contentBuilder.append(skel.heading)
                contentBuilder.append('\n')
                for child in skel.children:
                    annotated_content(child, contentBuilder)
                # contentBuilder.append('\n')

            elif isinstance(skel, List):
                annotated_content(skel.body, contentBuilder)

            elif isinstance(skel, Para):
                for body in skel.paragraph.bodies:
                    annotated_content_bodies(body, contentBuilder)
                contentBuilder.append('\n')
            else:
                pass

    def annotated_content_bodies(body, contentBuilder):
        if isinstance(body, ParaLink):
            contentBuilder.append(body.get_text(), body)

        elif isinstance(body, ParaText):
            contentBuilder.append(body.get_text())

        else:
            pass

    contentBuilder = AnnotatedContentBuilder()
    for child in section.children:
        annotated_content(child, contentBuilder)
    return contentBuilder





if __name__ == '__main__':

    import sys

    if len(sys.argv)<1 or len(sys.argv)>3:
        print("usage ",sys.argv[0]," articlefile")
        exit()

    articles=sys.argv[1]



    with open(articles, 'rb') as f:
        for p in iter_pages(f):
            print('\npagename:', p.page_name)
            print('\npageid:', p.page_id)


            print("get content of top-level sections, with subsections inlined and broken out entity offsets")
            for section in p.child_sections:

                print(" == ",section.heading ," ==")

                builder = annotate_section_content(section)
                print(builder.get_content())
                for ann in builder.get_annotations():
                    print(ann.start, ann.end, ann.annotation)

                print()
