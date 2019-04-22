package TextClassifier;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;

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

public class kNN{
	
	static Tokenizer tokenizer;
	
	int k_Near;
	int l_Vct;
	
	public kNN(int randvct,int nearest) throws InvalidFormatException, FileNotFoundException, IOException {
		tokenizer =new TokenizerME(new TokenizerModel(new FileInputStream("./data/Model/en-token.bin")));
		k_Near=nearest;
		l_Vct=randvct;
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
	
	public void kNNTestDoc(JSONObject json, HashSet<String> corpus,
			HashSet<String> stopwords, HashMap<String,Double> idf,
			ArrayList<Post> reviews, Double[][]RV) {
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
					for(int k=0;k<unigram.length;k++) {
						String token=unigram[k];
						token=SnowballStemming(Normalization(token));
						if(stopwords.contains(token)) {
							token=token.replaceAll("\\w+", "");
						}
						unigram[k]=token;
					}
					for(String token:unigram) {
						if(m_vector.containsKey(token)){
							m_vector.put(token, m_vector.get(token)+1);
						}
						else{
							m_vector.put(token,(double) 1);
						}
					}
					for(String token:idf.keySet()) {
						if(m_vector.containsKey(token)) {
							weight=(1+Math.log10(m_vector.get(token)))*idf.get(token);
							m_vector.put(token,weight);
						}
						else{
							m_vector.put(token,(double) 0);
						}
					}
					int[] hash=RandomProjection(m_vector,idf.keySet(),RV);
					review.setHash(hash);
					reviews.add(review);
				}
			}
		} catch (JSONException e) {
			e.printStackTrace();
		}
	}
	
	public void kNNClassifier(JSONObject json,HashSet<String> corpus,
			HashSet<String> stopwords, HashMap<String,Double> idf,
			ArrayList<Post> reviews, Double[][]RV,Integer[] count) {
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
					for(String token:idf.keySet()) {
						if(m_vector.containsKey(token)) {
							weight=(1+Math.log10(m_vector.get(token)))*idf.get(token);
							m_vector.put(token,weight);
						}
						else{
							m_vector.put(token,(double) 0);
						}
					}
					int[] h=RandomProjection(m_vector,idf.keySet(),RV);
					int tested=0;
					for(Post p:reviews) {
						if(Arrays.equals(h,p.getHash())&&count[tested]<k_Near){
							//if(rate>=4.0)
							if(help.getDouble(1)!=0&&help.getDouble(0)/help.getDouble(1)>=0.5)
								p.setVote((short)(p.getVote()+1));
							count[tested]++;
						}
						tested++;
					}
				}
			}
		} catch (JSONException e) {
			e.printStackTrace();
		}
	}
	
	public Double[][] generateRandom(){
		Double[][] R=new Double[l_Vct][5000];
		for(int i=0;i<l_Vct;i++) {
			for(int j=0;j<5000;j++) {
				R[i][j]=1-Math.random()*2;
			}
		}
		return R;
	}
	
	private int[] RandomProjection(HashMap<String, Double> featureVct, Set<String> feature, Double[][] R) {
		int[] hash = new int[l_Vct];
		for(int i=0;i<l_Vct;i++) {
			int j=0;
			double innerproduct=0;
			for(String K:feature) {
				innerproduct+=featureVct.get(K)*R[i][j];
				j++;
			}
			if(innerproduct>=0)
				hash[i]=1;
			else
				hash[i]=0;
		}
		return hash;
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