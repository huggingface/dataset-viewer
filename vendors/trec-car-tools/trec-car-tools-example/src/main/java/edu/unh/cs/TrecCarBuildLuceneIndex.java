package edu.unh.cs;

import edu.unh.cs.treccar_v2.Data;
import edu.unh.cs.treccar_v2.read_data.CborFileTypeException;
import edu.unh.cs.treccar_v2.read_data.CborRuntimeException;
import edu.unh.cs.treccar_v2.read_data.DeserializeData;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.*;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.jetbrains.annotations.NotNull;

import java.io.*;
import java.nio.file.FileSystems;
import java.nio.file.Path;
import java.util.Collections;
import java.util.Iterator;

/*
 * User: dietz
 * Date: 1/4/18
 * Time: 1:23 PM
 */

/**
 * Example of how to build a lucene index of trec car paragraphs
 */
public class TrecCarBuildLuceneIndex {

    private static void usage() {
        System.out.println("Command line parameters: paragraphs paragraphCBOR LuceneINDEX");
        System.exit(-1);
    }

    public static void main(String[] args) throws IOException {
        System.setProperty("file.encoding", "UTF-8");

        if (args.length < 3)
            usage();

        String mode = args[0];
        String indexPath = args[2];

        if (mode.equals("paragraphs")) {
            final String paragraphsFile = args[1];
            final FileInputStream fileInputStream2 = new FileInputStream(new File(paragraphsFile));

            System.out.println("Creating paragraph index in "+indexPath);
            final IndexWriter indexWriter = setupIndexWriter(indexPath, "paragraph.lucene");
            final Iterator<Data.Paragraph> paragraphIterator = DeserializeData.iterParagraphs(fileInputStream2);

            for (int i=1; paragraphIterator.hasNext(); i++){
                final Document doc = paragraphToLuceneDoc(paragraphIterator.next());
                indexWriter.addDocument(doc);
                if (i % 10000 == 0) {
                    System.out.print('.');
                    indexWriter.commit();
                }
            }

            System.out.println("\n Done indexing.");

            indexWriter.commit();
            indexWriter.close();
        }
        else if (mode.equals("pages")) {
            final String pagesFile = args[1];
            final FileInputStream fileInputStream = new FileInputStream(new File(pagesFile));

            System.out.println("Creating page index in "+indexPath);
            final IndexWriter indexWriter = setupIndexWriter(indexPath, "pages.lucene");

            final Iterator<Data.Page> pageIterator = DeserializeData.iterAnnotations(fileInputStream);

            for (int i=1; pageIterator.hasNext(); i++){
                final Document doc = pageToLuceneDoc(pageIterator.next());
                indexWriter.addDocument(doc);
                if (i % 10000 == 0) {
                    System.out.print('.');
                    indexWriter.commit();
                }
            }

            System.out.println("\n Done indexing.");


            indexWriter.commit();
            indexWriter.close();
        }
    }

    private static Iterable<Document> toIterable(final Iterator<Document> iter) throws CborRuntimeException, CborFileTypeException {
        return new Iterable<Document>() {
            @Override
            @NotNull
            public Iterator<Document> iterator() {
                return iter;
            }
        };
    }


    public static class ParaToLuceneIterator implements Iterator<Document> {
        private static final int DEBUG_EVERY = 10000;
        private int counter = DEBUG_EVERY;
        private final Iterator<Data.Paragraph> paragraphIterator;

        ParaToLuceneIterator(Iterator<Data.Paragraph> paragraphIterator){
            this.paragraphIterator = paragraphIterator;
        }

        @Override
        public boolean hasNext() {
            return this.paragraphIterator.hasNext();
        }

        @Override
        public Document next() {
            counter --;
            if(counter < 0) {
                System.out.print('.');
                counter = DEBUG_EVERY;
            }

            Data.Paragraph p = this.paragraphIterator.next();
            return paragraphToLuceneDoc(p);
        }

        @Override
        public void remove() {
            this.paragraphIterator.remove();
        }
    }

    @NotNull
    private static Document paragraphToLuceneDoc(Data.Paragraph p) {
        final Document doc = new Document();
        final String content = p.getTextOnly(); // <-- Todo Adapt this to your needs!
        doc.add(new TextField("text", content, Field.Store.YES));
        doc.add(new StringField("paragraphid", p.getParaId(), Field.Store.YES));  // don't tokenize this!
        return doc;
    }


    public static class PageToLuceneIterator implements Iterator<Document> {
        private static final int DEBUG_EVERY = 1000;
        private int counter = DEBUG_EVERY;
        private final Iterator<Data.Page> pageIterator;

        PageToLuceneIterator(Iterator<Data.Page> pageIterator){
            this.pageIterator = pageIterator;
        }

        @Override
        public boolean hasNext() {
            return this.pageIterator.hasNext();
        }

        @Override
        public Document next() {
            counter --;
            if(counter < 0) {
                System.out.print('.');
                counter = DEBUG_EVERY;
            }

            Data.Page p = this.pageIterator.next();
            return pageToLuceneDoc(p);
        }

        @Override
        public void remove() {
            this.pageIterator.remove();
        }
    }

    @NotNull
    private static Document pageToLuceneDoc(Data.Page p) {
        final Document doc = new Document();
        StringBuilder content = new StringBuilder();
        pageContent(p, content);                    // Todo Adapt this to your needs!

        doc.add(new TextField("text",  content.toString(), Field.Store.NO));  // dont store, just index
        doc.add(new StringField("pageid", p.getPageId(), Field.Store.YES));  // don't tokenize this!
        return doc;
    }


    private static void sectionContent(Data.Section section, StringBuilder content){
        content.append(section.getHeading()+'\n');
        for (Data.PageSkeleton skel: section.getChildren()) {
            if (skel instanceof Data.Section) sectionContent((Data.Section) skel, content);
            else if (skel instanceof Data.Para) paragraphContent((Data.Para) skel, content);
            else {
            }
        }
    }
    private static void paragraphContent(Data.Para paragraph, StringBuilder content){
        content.append(paragraph.getParagraph().getTextOnly()).append('\n');
    }
    private static void pageContent(Data.Page page, StringBuilder content){
        content.append(page.getPageName()).append('\n');

        for(Data.PageSkeleton skel: page.getSkeleton()){
            if(skel instanceof Data.Section) sectionContent((Data.Section) skel, content);
            else if(skel instanceof Data.Para) paragraphContent((Data.Para) skel, content);
            else {}    // ignore other
        }

    }
    @NotNull
    private static IndexWriter setupIndexWriter(String indexPath, String typeIndex) throws IOException {
        Path path = FileSystems.getDefault().getPath(indexPath, typeIndex);
        Directory indexDir = FSDirectory.open(path);
        IndexWriterConfig config = new IndexWriterConfig(new StandardAnalyzer());
        return new IndexWriter(indexDir, config);
    }
}