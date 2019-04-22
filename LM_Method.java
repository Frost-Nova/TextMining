package VSM_LM;

import java.util.ArrayList;
import java.util.HashMap;

import org.tartarus.snowball.SnowballStemmer;
import org.tartarus.snowball.ext.englishStemmer;

import json.JSONArray;
import json.JSONException;
import json.JSONObject;
import opennlp.tools.tokenize.Tokenizer;
import structures.LanguageModel;
import structures.Post;
import structures.Token;

public class LM_Method{
	
	private static Tokenizer tokenizer;
	
	private static String[] Tokenize(String text) {
		return tokenizer.tokenize(text);
	}
	
	public static void evaluateLM(JSONObject json,ArrayList<Post> reviews) {
		try {
			JSONArray jarray = json.getJSONArray("Reviews");
			for(int i=0; i<jarray.length(); i++) {
				Post review = new Post(jarray.getJSONObject(i));
				//Double weight;
				HashMap<String, Double> m_vector = new HashMap<String, Double>();
				String[] unigram = Tokenize(review.getContent());
				ArrayList<String> tokens = new ArrayList<String>();
				String[] bigram = new String[unigram.length-1];
				for(int k=0;k<unigram.length;k++) {
					String token=unigram[k];
					unigram[k]=SnowballStemming(Normalization(token));
					//if(!token.isEmpty())
					//	tokens.add(token);
				}
				for(int j=0;j<unigram.length-1;j++) {
					bigram[j]=unigram[j]+"-"+unigram[j+1];
					if(!unigram[j].isEmpty())
						tokens.add(bigram[j]);
				}
				for(String token:tokens) {
					//if(Arrays.asList(m_stopwords.toArray()).contains(token)) {
						//token=token.replaceAll("\\w+", "");
					//}
					if(m_vector.containsKey(token)){
						m_vector.put(token,m_vector.get(token)+1);
					}
					else{
						m_vector.put(token,(double)1);
					}
				}
				/*for(String token:m_stats.keySet()) {
					if(m_vector.containsKey(token)) {
						weight=(1+Math.log10(m_vector.get(token).getValue()))*m_stats.get(token).getValue();
						m_vector.get(token).setValue(weight);
					}
					else if(!token.isEmpty()){
						Token T = new Token(token);
						m_vector.put(T.getToken(),T);
					}
				}*/
				review.setVct(m_vector);
				reviews.add(review);
				/*if(review.getID().equals("nQ28SMlwUcpJ_1CgpOV1pA")) {
					System.out.format("%s\n%s\n%s\n", review.getAuthor(),review.getContent(),review.getDate());
				}*/
			}
		} catch (JSONException e) {
			e.printStackTrace();
		}
	}

	
	public void createLanguageModel() {
		
	}
	
	public static void generateSentence(LanguageModel L1,LanguageModel L2) {
		double prob=1.0;
		String sentence="";
		String word="";
		String first="";
		
		for(int i=0;i<10;i++) {
			first=L1.Tagsampling(null);
			sentence+=(first+" ");
			prob*=L1.calcLinearSmoothedProb(first);
			for(int j=0;j<14;j++) {
				word=L2.Tagsampling(first);
				if(!(word==null)) {
					first=word.split("-")[1];
					sentence+=(first+" ");
					prob*=L2.calcAbsoluteDiscountProb(word);
				}
				else {
					first=L1.Tagsampling(null);
					sentence+=(first+" ");
					prob*=L1.calcLinearSmoothedProb(first);
				}
			}
			System.out.format("%s\t%s\n",sentence,prob);
			sentence="";
			prob=1.0;
		}
	}
	
	public static void calcPP(LanguageModel Model,ArrayList<Post> reviews) {
		double PP=0;
		for(Post review:reviews) {
			//String[] tokens = review.getTokens();
			if(review.getTokens().length!=0) {
				PP=Math.exp(-Model.logLikelihood(review)/review.getTokens().length);
			}
			System.out.format("%s\n",PP);
		}
		
		/*for(int j=0;j<m_reviews.size();j++) {
			sigma+=Math.pow((PP[j]-avg), 2);
		}
		sigma=Math.sqrt(sigma/m_reviews.size());*/
	}

	private static String SnowballStemming(String token) {
		SnowballStemmer stemmer = new englishStemmer();
		stemmer.setCurrent(token);
		if (stemmer.stem())
			return stemmer.getCurrent();
		else
			return token;
	}
	
	private static String Normalization(String token) {
		// remove all non-word characters
		// please change this to removing all English punctuation
		token = token.replaceAll("\\p{Punct}", ""); 
		// convert to lower case
		token = token.toLowerCase(); 
		// add a line to recognize integers and doubles via regular expression
		// and convert the recognized integers and doubles to a special symbol "NUM"
		token = token.replaceAll("\\d+", "NUM");
		return token;
	}
	
}