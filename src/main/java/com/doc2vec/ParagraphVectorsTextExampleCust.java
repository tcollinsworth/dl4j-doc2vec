package com.doc2vec;

import java.io.File;
import java.util.Collection;

import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.AbstractCache;
import org.deeplearning4j.text.documentiterator.LabelsSource;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * This is example code for dl4j ParagraphVectors implementation. In this example we build distributed representation of
 * all sentences present in training corpus. However, you still use it for training on labelled documents, using sets of
 * LabelledDocument and LabelAwareIterator implementation.
 *
 * ************************************************************************************************* PLEASE NOTE: THIS
 * EXAMPLE REQUIRES DL4J/ND4J VERSIONS >= rc3.8 TO COMPILE SUCCESSFULLY
 * *************************************************************************************************
 *
 * @author raver119@gmail.com
 */
public class ParagraphVectorsTextExampleCust {
	private static final Logger log = LoggerFactory.getLogger(ParagraphVectorsTextExample.class);

	public static void main(String[] args) throws Exception {
		ClassPathResource resource = new ClassPathResource("/raw_sentences.txt");
		File file = resource.getFile();
		SentenceIterator iter = new BasicLineIterator(file);
		AbstractCache<VocabWord> cache = new AbstractCache<>();
		TokenizerFactory t = new DefaultTokenizerFactory();
		t.setTokenPreProcessor(new CommonPreprocessor());
		/*
		 * if you don't have LabelAwareIterator handy, you can use synchronized labels generator it will be used to
		 * label each document/sequence/line with it's own label. But if you have LabelAwareIterator ready, you can can
		 * provide it, for your in-house labels
		 */
		LabelsSource source = new LabelsSource("DOC_");

		ParagraphVectors vec = new ParagraphVectors.Builder().minWordFrequency(1).iterations(10).epochs(1)
				.layerSize(50).learningRate(0.015).labelsSource(source).windowSize(5).iterate(iter)
				.trainWordVectors(false).vocabCache(cache).tokenizerFactory(t).sampling(0).build();
		vec.fit();
		/*
		 * In training corpus we have few lines that contain pretty close words involved. These sentences should be
		 * pretty close to each other in vector space line 3721: This is my way . line 6348: This is my case . line
		 * 9836: This is my house . line 12493: This is my world . line 16393: This is my work . this is special
		 * sentence, that has nothing common with previous sentences line 9853: We now have one . Note that docs are
		 * indexed from 0
		 */
		Collection<String> collection = vec.predictSeveral("the dog and the fox .", 5);
		for (String docId : collection) {
			double similarity = vec.similarity("DOC_97163", docId);
			log.info("DOC_97163 / " + docId + " similarity: " + similarity);
		}
	}
}
