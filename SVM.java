package TextClassifier;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.HashMap;
import java.util.HashSet;

import org.tartarus.snowball.SnowballStemmer;
import org.tartarus.snowball.ext.englishStemmer;

import json.JSONArray;
import json.JSONException;
import json.JSONObject;
import opennlp.tools.tokenize.Tokenizer;
import opennlp.tools.tokenize.TokenizerME;
import opennlp.tools.tokenize.TokenizerModel;
import opennlp.tools.util.InvalidFormatException;
import structures.Post;

public class SVM{
	static Tokenizer tokenizer;
	
	public SVM() throws InvalidFormatException, FileNotFoundException, IOException {
		tokenizer =new TokenizerME(new TokenizerModel(new FileInputStream("./data/Model/en-token.bin")));
	}
	
	private static String[] Tokenize(String text) {
		return tokenizer.tokenize(text);
	}
	
	public void calcDF(JSONObject json,HashSet<String> corpus,HashSet<String> stopwords,
			HashSet<String> features,HashMap<String,Double> docfre) {
		try {
			JSONArray jarray = json.getJSONArray("Reviews");
			for(int i=0; i<jarray.length(); i++) {
				Post review = new Post(jarray.getJSONObject(i));
				if(review.isEmpty())
					continue;
				String ID = review.getID();
				if(corpus.contains(ID)) {
					String[] unigram = Tokenize(review.getContent());
					HashSet<String> unique = new HashSet<>();	
					for(String token:unigram) {
						token=SnowballStemming(Normalization(token));
						if(stopwords.contains(token)) {
							token=token.replaceAll("\\w+", "");
						}
						if(!token.isEmpty())
							unique.add(token);
					}
					for(String word:features) {
						if(unique.contains(word)&&docfre.containsKey(word))
							docfre.put(word,docfre.get(word)+1);
						else if(unique.contains(word)&&!docfre.containsKey(word))
							docfre.put(word, (double) 1);
					}
				}
			}
		} catch (JSONException e) {
			e.printStackTrace();
		}
	}
	
	public void SVM_VSM(JSONObject json,HashSet<String> corpus,HashSet<String> stopwords,
			HashMap<String,Double> idf, PrintWriter write) {
		try {
			JSONArray jarray = json.getJSONArray("Reviews");
			for(int i=0; i<jarray.length(); i++) {
				Post review = new Post(jarray.getJSONObject(i));
				if(review.isEmpty())
					continue;
				String ID = review.getID();			
				if(corpus.contains(ID)) {
					HashMap<String, Double> m_vector = new HashMap<String, Double>();
					String[] unigram = Tokenize(review.getContent());
					double weight=0;
					//double rate = review.getRating();
					JSONArray help=review.getHelp();
					int ybar=0;
					//if(rate>=4.0) {
					if(help.getDouble(1)!=0&&help.getDouble(0)/help.getDouble(1)>=0.5) {
						ybar=1;					
					}
					write.print(ybar+" ");
					for(int k=0;k<unigram.length;k++) {
						String token=unigram[k];
						token=SnowballStemming(Normalization(token));
						if(stopwords.contains(token)) {
							token=token.replaceAll("\\w+", "");
						}
						unigram[k]=token;
					}
					for(String token:unigram) {
						if(idf.keySet().contains(token)) {
							if(m_vector.containsKey(token)){
								m_vector.put(token, m_vector.get(token)+1);
							}
							else{
								m_vector.put(token,(double) 1);
							}
						}
					}
					int j=0;
					for(String token:idf.keySet()) {
						if(m_vector.containsKey(token)) {
							weight=(1+Math.log10(m_vector.get(token)))*idf.get(token);
							m_vector.put(token,weight);
						}
						else{
							m_vector.put(token,(double) 0);
						}
						j++;
						write.print(j+":"+m_vector.get(token)+" ");	
					}
					write.print("\n");
				}
			}
		} catch (JSONException e) {
			e.printStackTrace();
		}
	}
	
	private static String Normalization(String token) {
		token = token.replaceAll("\\p{Punct}", "");
		token = token.toLowerCase();		
		token = token.replaceAll("\\d+", "NUM");
		return token;
	}
	
	private static String SnowballStemming(String token) {
		SnowballStemmer stemmer = new englishStemmer();
		stemmer.setCurrent(token);
		if (stemmer.stem())
			return stemmer.getCurrent();
		else
			return token;
	}
}