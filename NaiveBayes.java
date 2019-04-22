package TextClassifier;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;
import java.util.TreeMap;

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

public class NaiveBayes{
	
	static Tokenizer tokenizer;
	
	public NaiveBayes() throws InvalidFormatException, FileNotFoundException, IOException{
		tokenizer =new TokenizerME(new TokenizerModel(new FileInputStream("./data/Model/en-token.bin")));
	}
	
	private static String[] Tokenize(String text) {
		return tokenizer.tokenize(text);
	}
	
	public void NBParameter(JSONObject json,HashSet<String> corpus,
			HashSet<String> stopwords,HashSet<String> features,Integer[] docs,
			HashMap<String, Double> PosYes,//HashMap<String, Integer> PosNo,HashMap<String, Integer> NegNo,
			HashMap<String, Double> NegYes){
		try {
			JSONArray jarray = json.getJSONArray("Reviews");
			for(int i=0; i<jarray.length(); i++) {
				Post review = new Post(jarray.getJSONObject(i));
				if(review.isEmpty())
					continue;
				String ID = review.getID();
				if(corpus.contains(ID)) {
					String[] unigram = Tokenize(review.getContent());
					double rate = review.getRating();
					//JSONArray help=review.getHelp();
					HashSet<String> unique = new HashSet<>();
					for(String token:unigram) {
						token=SnowballStemming(Normalization(token));
						if(Arrays.asList(stopwords.toArray()).contains(token)) {
							token=token.replaceAll("\\w+", "");
						}
						if(!token.isEmpty())
							unique.add(token);
					}
					if(rate>=4.0)
					//if(help.getDouble(1)!=0&&help.getDouble(0)/help.getDouble(1)>=0.5)
						docs[0]++;
					else
						docs[1]++;
					for(String word:features) {
						if(unique.contains(word)) {
							if(rate>=4.0){
							//if(help.getDouble(1)!=0&&help.getDouble(0)/help.getDouble(1)>=0.5){
								if(PosYes.containsKey(word))
									PosYes.put(word, PosYes.get(word)+1);
								else
									PosYes.put(word, (double) 1);
							}else {
								if(NegYes.containsKey(word))
									NegYes.put(word, NegYes.get(word)+1);
								else
									NegYes.put(word, (double) 1);
							}
						}/*else {
							if(rate>=4.0) {
								if(PosNo.containsKey(word))
									PosNo.put(word, PosNo.get(word)+1);
								else
									PosNo.put(word, 1);
							}else {
								if(NegNo.containsKey(word))
									NegNo.put(word, NegNo.get(word)+1);
								else
									NegNo.put(word, 1);
							}
						}*/
					}
					//System.out.println(posdoc+"\t"+negdoc);
				}
			}
		} catch (JSONException e) {
			e.printStackTrace();
		}
	}
	
	public static HashMap<String,Double> calcFeature(int posdoc,int negdoc,HashSet<String> features,
			HashMap<String, Integer> PosYes,HashMap<String, Integer> PosNo,
			HashMap<String, Integer> NegYes,HashMap<String, Integer> NegNo)
	{	
		HashMap<String,Double>	stats=new HashMap<>();
		double Ppos=(double)posdoc/(posdoc+negdoc);
		double Pneg=(double)negdoc/(posdoc+negdoc);
		double Classp=Ppos*Math.log(Ppos)+Pneg*Math.log(Pneg);
		for(String word:features) {
			long PY=0,PN=0,NY=0,NN=0;
			double Chi,IG,Epy=0,Eny=0,Epn=0,Enn=0;
			if(PosYes.containsKey(word))
				PY=PosYes.get(word);
			if(PosNo.containsKey(word))
				PN=PosNo.get(word);
			if(NegYes.containsKey(word))
				NY=NegYes.get(word);
			if(NegNo.containsKey(word))
				NN=NegNo.get(word);
			if(PosYes.containsKey(word))
				Epy=(double)PY/(PY+NY)*Math.log((double)PY/(PY+NY));
			if(PosNo.containsKey(word))
				Epn=(double)PN/(NN+PN)*Math.log((double)PN/(NN+PN));
			if(NegYes.containsKey(word))
				Eny=(double)NY/(PY+NY)*Math.log((double)NY/(PY+NY));
			if(NegNo.containsKey(word))
				Enn=(double)NN/(NN+PN)*Math.log((double)NN/(NN+PN));
			
			IG=-Classp+(double)(PY+NY)/(posdoc+negdoc)*(Epy+Eny)+(double)(PN+NN)/(posdoc+negdoc)*(Epn+Enn);
			
			//Chi=(PY+PN+NY+NN)*Math.pow((PY*NN-PN*NY),2)/((PY+NY)*(PN+NN)*(PY+PN)*(NY+NN));
			stats.put(word,IG);
		}
		return stats;
	}
	
	public void NBTrain(int posdoc,int negdoc,HashSet<String> features,
			HashMap<String, Double> PosYes,HashMap<String, Double> NegYes)
	{
		for(String word:features) {
			if(PosYes.keySet().contains(word))
				PosYes.put(word, (PosYes.get(word)+0.1)/posdoc);
			else
				PosYes.put(word,0.1/posdoc);
			if(NegYes.keySet().contains(word))
				NegYes.put(word, (NegYes.get(word)+0.1)/negdoc);
			else
				NegYes.put(word,0.1/negdoc);
			//m_stats.put(word,Math.log(PosYes.get(word)/NegYes.get(word)));
			//System.out.println(word+"\t"+m_stats.get(word));
		}
	}
	
	public void NBClassifier(JSONObject json,int posdoc,int negdoc,
			HashSet<String> corpus,HashSet<String> stopwords,HashSet<String> features,
			HashMap<String, Double> PosYes,HashMap<String, Double> NegYes,Integer[] re)//HashMap<Double,Integer> stats)
	{
		try {
			JSONArray jarray = json.getJSONArray("Reviews");
			for(int i=0; i<jarray.length(); i++) {
				Post review = new Post(jarray.getJSONObject(i));
				if(review.isEmpty())
					continue;
				String ID = review.getID();
				if(corpus.contains(ID)) {
					String[] unigram = Tokenize(review.getContent());
					double rate = review.getRating();
					//JSONArray help=review.getHelp();
					HashSet<String> unique = new HashSet<>();
					double fX=Math.log((double)posdoc/negdoc);
					int ybar=0;
					int ypre=0;
					if(rate>=4.0) {
					//if(help.getDouble(1)!=0&&help.getDouble(0)/help.getDouble(1)>=0.5) {
						ybar=1;re[1]++;						
					}
					for(String token:unigram) {
						token=SnowballStemming(Normalization(token));
						if(Arrays.asList(stopwords.toArray()).contains(token)) {
							token=token.replaceAll("\\w+", "");
						}
						if(!token.isEmpty())
							unique.add(token);
					}
					for(String word:unique) {
						if(features.contains(word))
							fX=fX+Math.log(PosYes.get(word))-Math.log(NegYes.get(word));
					}
					//stats.put(fX, ybar);
					if(fX>=-3.0) {
						ypre=1;re[2]++;						
					}
					if(ypre==1&&ybar==1)
						re[0]++;
				}
			}
		}catch (JSONException e) {
			e.printStackTrace();
		}
	}
	
	public static void PRCurve(int posdoc,HashMap<Double,Integer> stats){
		int TP=0,i=0;
		double[][] PR=new double[stats.size()][2];
		Map<Double, Integer> treeMap=new TreeMap<>(Collections.reverseOrder());
		treeMap.putAll(stats);
		
		Iterator<Map.Entry<Double, Integer>> it=treeMap.entrySet().iterator();
		while(it.hasNext()) {
			Map.Entry<Double, Integer> entry=it.next();
			i++;			
			if(entry.getValue()==1)
				TP++;
			PR[i-1][0]=(double)TP/i;
			PR[i-1][1]=(double)TP/posdoc;
			if(i%10==0)
				System.out.println(PR[i-1][0]+"\t"+PR[i-1][1]);
		}
		File file = new File("./data/PRCurve_PN.txt");
		PrintWriter writer;
		try {
			writer = new PrintWriter(file);
			for(int j=0;j<stats.size();j++) {
				writer.println(PR[j][0]+"\t"+PR[j][1]);
			}
			writer.close();
		} catch (FileNotFoundException e) {
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