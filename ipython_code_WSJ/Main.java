import edu.stanford.nlp.ie.util.RelationTriple;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.naturalli.NaturalLogicAnnotations;
import edu.stanford.nlp.util.CoreMap;

import java.io.*;
import java.util.Collection;
import java.util.List;
import java.util.Properties;

/**
 * A demo illustrating how to call the OpenIE system programmatically.
 */
public class Main {

    public static void main(String[] args) throws Exception {
        // Create the Stanford CoreNLP pipeline
        Properties props = new Properties();
        props.setProperty("annotators", "tokenize,ssplit,pos,lemma,depparse,natlog,openie");
        StanfordCoreNLP pipeline = new StanfordCoreNLP(props);

        // Annotate an example document.
        String file = "input/billip_num_percent_sent";
        //String file = "input/test_sent.txt";
        String out_put_file = "result/triples";
        BufferedReader br = new BufferedReader(new FileReader(file));
        BufferedWriter wr = new BufferedWriter(new FileWriter(out_put_file));
        String test_sentence = null;
        Integer line_number = 0;
        while ((test_sentence = br.readLine())!= null){
            line_number ++;
           // test_sentence = "MacMillan Bloedel Ltd. said it plans to redeem all of its 9 %, Series J debentures outstanding April 27. ";
            Annotation doc = new Annotation(test_sentence);
            pipeline.annotate(doc);
            // Loop over sentences in the document
            for (CoreMap sentence : doc.get(CoreAnnotations.SentencesAnnotation.class)) {
                // Get the OpenIE triples for the sentence
                Collection<RelationTriple> triples = sentence.get(NaturalLogicAnnotations.RelationTriplesAnnotation.class);
                // Print the triples
                for (RelationTriple triple : triples) {
                    if (triple.confidence > 0.9){
                        if (triple.objectGloss().contains("%") || triple.objectGloss().contains("percent")){
                            wr.write(line_number.toString()+", ");
                            wr.write(triple.subjectLemmaGloss()+", ");
                            wr.write(triple.relationLemmaGloss()+", ");
                            wr.write(triple.objectLemmaGloss()+"\n");
                        }

                     }

                }
            }
        }
    br.close();
    wr.close();
    }
}