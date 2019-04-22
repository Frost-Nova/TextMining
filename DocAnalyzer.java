/**
 * 
 */
package analyzer;

import java.io.*;
import java.time.Duration;
import java.time.LocalDateTime;
import java.util.*;
import org.tartarus.snowball.SnowballStemmer;
import org.tartarus.snowball.ext.englishStemmer;
import org.tartarus.snowball.ext.porterStemmer;

import TextClassifier.Methods;
import TextClassifier.NaiveBayes;
import TextClassifier.SVM;
import TextClassifier.kNN;
import VSM_LM.VSM_Method;
import json.JSONArray;
import json.JSONException;
import json.JSONObject;
import opennlp.tools.tokenize.Tokenizer;
import opennlp.tools.tokenize.TokenizerME;
import opennlp.tools.tokenize.TokenizerModel;
import opennlp.tools.util.InvalidFormatException;
import structures.Post;
import structures.Token;

/**
 * @author hongning
 * Sample codes for demonstrating OpenNLP package usage
 * NOTE: the code here is only for demonstration purpose,
 * please revise it accordingly to maximize your implementation's efficiency!
 */
public class DocAnalyzer {
	//N-gram to be created
	int m_N;
	int posdoc;
	int negdoc;
	HashSet<String> m_stopwords;
	HashSet<String> corpus_train;
	HashSet<String> corpus_test;
	HashSet<String> features;
	//double del;double sig;

	ArrayList<Post> m_reviews;
	
	HashMap<Double,Integer> m_stats;

	//HashMap<String, List<String>> Prox;
	HashMap<String, Double> posiyes;
	HashMap<String, Double> negyes;
	
	HashMap<String, Integer> PosYes;
	HashMap<String, Integer> NegYes;
	HashMap<String, Integer> PosNo;
	HashMap<String, Integer> NegNo;
	
	Tokenizer m_tokenizer;
	
	//LanguageModel unigramModel;
	//LanguageModel wordtagModel;
	//LanguageModel tagtagModel;
	
	//PrintWriter writer;
	
	public DocAnalyzer(String tokenModel, int N) throws InvalidFormatException, FileNotFoundException, IOException {
		m_N = N;
		m_tokenizer = new TokenizerME(new TokenizerModel(new FileInputStream(tokenModel)));
		m_reviews = new ArrayList<Post>();
		m_stopwords = new HashSet<String>();
		corpus_train = new HashSet<String>();
		corpus_test = new HashSet<String>();	
		features = new HashSet<String>();
		
		posiyes = new HashMap<String, Double>();
		negyes = new HashMap<String, Double>();
	
		//RandVct = new double[5][5000];
		//count = new int[5];
		posdoc=0;negdoc=0;
		//Prox=new HashMap<String, List<String>>();
		PosYes=new HashMap<String, Integer>();
		PosNo=new HashMap<String, Integer>();
		NegYes=new HashMap<String, Integer>();
		NegNo=new HashMap<String, Integer>();
		m_stats = new HashMap<Double,Integer>();
		//File file=new File("./data/AmazonNew.txt");
		//writer=new PrintWriter(file);
	}
	
	public void EvaluateNBClassifier(String folder, String suffix) throws InvalidFormatException, FileNotFoundException, IOException {
		NaiveBayes NB=new NaiveBayes();
		Integer[] Result = {0,0,0};
		Integer[] docount = {0,0};
		int posdoc=0, negdoc=0;
		int TP=0,AP=0,PP=0;
		double precision,recall,F1;
		File dir = new File(folder);
		for (File f : dir.listFiles()) {
			if (f.isFile() && f.getName().endsWith(suffix)) {
				NB.NBParameter(LoadJson(f.getAbsolutePath()),corpus_train,m_stopwords,
						features, docount, posiyes, negyes);
			}
		}
		posdoc=docount[0];negdoc=docount[1];
		System.out.println("Positive Document: "+posdoc+"\n"+"Negative Document: "+negdoc);
		NB.NBTrain(posdoc, negdoc, features, posiyes, negyes);
		for (File f : dir.listFiles()) {
			if (f.isFile() && f.getName().endsWith(suffix)) {
				NB.NBClassifier(LoadJson(f.getAbsolutePath()), posdoc, negdoc,
						corpus_test, m_stopwords, features, posiyes, negyes, Result);//m_stats);
			}
		}
		//NaiveBayes.PRCurve(posdoc, m_stats);
		TP=Result[0]; AP=Result[1]; PP=Result[2];
		//System.out.print("True positive: "+TP+"\n"+"All positive: "+AP+"\n"+"Predicted positve: "+PP);
		precision=(double)TP/PP;recall=(double)TP/AP;F1=2/(1/precision+1/recall);
		System.out.print("Precision: "+precision+"\n"+"Recall: "+recall+"\n"+"F1: "+F1);
	}

	public void EvaluatekNNClassifier(String folder, String suffix) throws InvalidFormatException, FileNotFoundException, IOException, JSONException {
		int l=7,k=11;
		kNN kNN55=new kNN(l,k);
		File dir = new File(folder);
		HashMap<String,Double> IDF= new HashMap<>();
		Double[][] RandVct;
		double idf,precision,recall,F1;
		int TP=0,AP=0,PP=0;
		for (File f : dir.listFiles()) {
			if (f.isFile() && f.getName().endsWith(suffix))
				kNN55.calcDF(LoadJson(f.getAbsolutePath()),corpus_train,m_stopwords,features, IDF);
		}
		System.out.println("Number of training set is: "+corpus_train.size());
		for(String T:features) {
			if(IDF.containsKey(T)) {
				idf=Math.log10(corpus_train.size()/IDF.get(T));
				IDF.put(T,idf);
			}else
				IDF.put(T, (double) 0);
		}
		System.out.println("IDF calculation finished!");
		RandVct=kNN55.generateRandom();
		for (File f : dir.listFiles()) {
			if (f.isFile() && f.getName().endsWith(suffix))
				kNN55.kNNTestDoc(LoadJson(f.getAbsolutePath()),corpus_test,m_stopwords,
					IDF, m_reviews,RandVct);
		}
		System.out.println("Test set calculation finished with "+m_reviews.size()+" Documents.");
		Integer[] Count = new Integer[m_reviews.size()];
		Integer[] Goal=new Integer[m_reviews.size()];
		for(int i=0;i<Goal.length;i++) {
			Goal[i]=k;
			Count[i]=0;
		}
		for (File f : dir.listFiles()) {
			if (f.isFile() && f.getName().endsWith(suffix))
				kNN55.kNNClassifier(LoadJson(f.getAbsolutePath()),corpus_train,m_stopwords,
					IDF, m_reviews,RandVct,Count);
			if(Arrays.equals(Count, Goal))
				break;
		}
		for(Post P:m_reviews) {
			//if(P.getRating()>=4.0) {
			if(P.getHelp().getDouble(1)!=0&&P.getHelp().getDouble(0)/P.getHelp().getDouble(1)>=0.5){
				if(P.getVote()>k/2)
					TP++;
				AP++;
			}
			if(P.getVote()>k/2)
				PP++;
		}
		precision=(double)TP/PP;recall=(double)TP/AP;F1=2/(1/precision+1/recall);
		System.out.print("Precision: "+precision+"\n"+"Recall: "+recall+"\n"+"F1: "+F1);
	}
	
	public void SVMClassifier(String folder, String suffix) throws InvalidFormatException, FileNotFoundException, IOException {
		SVM svm=new SVM();
		File dir = new File(folder);
		HashMap<String,Double> IDF= new HashMap<>();
		double idf;
		for (File f : dir.listFiles()) {
			if (f.isFile() && f.getName().endsWith(suffix))
				svm.calcDF(LoadJson(f.getAbsolutePath()),corpus_train,m_stopwords,features, IDF);
		}
		for(String T:features) {
			if(IDF.containsKey(T)) {
				idf=Math.log10(corpus_train.size()/IDF.get(T));
				IDF.put(T,idf);
			}else
				IDF.put(T, (double) 0);
		}
		System.out.println("IDF calculation finished!");
		File file1=new File("./data/F5_Train.txt");
		PrintWriter writer1=new PrintWriter(file1);
		File file2=new File("./data/F5_Test.txt");
		PrintWriter writer2=new PrintWriter(file2);
		for (File f : dir.listFiles()) {
			if (f.isFile() && f.getName().endsWith(suffix))
				svm.SVM_VSM(LoadJson(f.getAbsolutePath()),corpus_train,m_stopwords,IDF,writer1);
		}
		System.out.println("Train set calculation finished!");
		for (File f : dir.listFiles()) {
			if (f.isFile() && f.getName().endsWith(suffix))
				svm.SVM_VSM(LoadJson(f.getAbsolutePath()),corpus_test,m_stopwords,IDF,writer2);
		}
		System.out.println("Test set calculation finished!");
		writer1.close();writer2.close();
	}
	
	//sample code for loading a json file
	public JSONObject LoadJson(String filename) {
		try {
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
			StringBuffer buffer = new StringBuffer(1024);
			String line;
			
			while((line=reader.readLine())!=null) {
				buffer.append(line);
			}
			reader.close();
			
			return new JSONObject(buffer.toString());
		} catch (IOException e) {
			System.err.format("[Error]Failed to open file %s!", filename);
			e.printStackTrace();
			return null;
		} catch (JSONException e) {
			System.err.format("[Error]Failed to parse json file %s!", filename);
			e.printStackTrace();
			return null;
		}
	}
	
	public String LoadText(String filename) {
		try {
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
			StringBuffer buffer = new StringBuffer(1024);
			String line;
			while((line=reader.readLine())!=null) {
					buffer.append(line);
			}
			reader.close();
			System.out.println("Loading finished!");
			return buffer.toString();
		} catch (IOException e) {
			System.err.format("[Error]Failed to open file %s!", filename);
			e.printStackTrace();
			return null;
		}
	}
	
	public void FeatureSel(JSONObject json,ArrayList<Post> reviews) {
		try {
			JSONArray jarray = json.getJSONArray("Reviews");
			//System.out.println(jarray.get(0));
			for(int i=0; i<jarray.length(); i++) {
				Post review = new Post(jarray.getJSONObject(i));
				if(review.isEmpty())
					continue;
				//JSONArray help=review.getHelp();
				//System.out.println(help.getDouble(0)+"\t"+help.getDouble(1));
				int count=0;
				String[] unigram=Tokenize(review.getContent());
				HashSet<String> unique = new HashSet<>();
				//double rate = review.getRating();
				for(String token:unigram) {
					token=SnowballStemming(Normalization(token));
					if(m_stopwords.contains(token)) {
						token=token.replaceAll("\\w+", "");
					}
					if(!token.isEmpty())
						unique.add(token);
				}
				for(String word:features) {
					if(unique.contains(word))
						count++;
				}
				if(count>5)
					corpus_train.add(review.getID());
			}
		} catch (JSONException e) {
			e.printStackTrace();
		}
	}
	
	// sample code for demonstrating how to recursively load files in a directory
	public void LoadDirectory(String folder, String suffix) {
		File dir = new File(folder);
		int size = m_reviews.size();
		for (File f : dir.listFiles()) {
			if (f.isFile() && f.getName().endsWith(suffix)) {
				FeatureSel(LoadJson(f.getAbsolutePath()),m_reviews);
			}
			else if (f.isDirectory())
				LoadDirectory(f.getAbsolutePath(), suffix);
		}
		size = m_reviews.size() - size;
		System.out.println("Analyzing " + size + " review documents from " + folder);
	}
	
	public void LoadTestDirectory(String folder, String suffix) {
		File dir = new File(folder);
		for (File f : dir.listFiles()) {
			if (f.isFile() && f.getName().endsWith(suffix))
				break;
				//analyzeTestDocument(LoadJson(f.getAbsolutePath()));
			//else if (f.isDirectory())
				//LoadTestDirectory(f.getAbsolutePath(), suffix);
		}
	}

	//sample code for demonstrating how to use Snowball stemmer
	public String SnowballStemming(String token) {
		SnowballStemmer stemmer = new englishStemmer();
		stemmer.setCurrent(token);
		if (stemmer.stem())
			return stemmer.getCurrent();
		else
			return token;
	}
	
	//sample code for demonstrating how to use Porter stemmer
	public String PorterStemming(String token) {
		porterStemmer stemmer = new porterStemmer();
		stemmer.setCurrent(token);
		if (stemmer.stem())
			return stemmer.getCurrent();
		else
			return token;
	}
	
	//sample code for demonstrating how to perform text normalization
	//you should implement your own normalization procedure here
	public String Normalization(String token) {
		token = token.replaceAll("\\p{Punct}", "");
		token = token.toLowerCase();		
		token = token.replaceAll("\\d+", "NUM");
		return token;
	}
	
	String[] Tokenize(String text) {
		return m_tokenizer.tokenize(text);
	}
	
	public void TokenizerDemon(String text) {
		System.out.format("Tokenization\tNormalization\tSnonball Stemming\n");
		for(String token:m_tokenizer.tokenize(text)){
			System.out.format("%s\t%s\t%s\n", token, Normalization(token), SnowballStemming(Normalization(token)));
		}
	}
	
	public static void main(String[] args) throws InvalidFormatException, FileNotFoundException, IOException, JSONException {		
		DocAnalyzer analyzer = new DocAnalyzer("./data/Model/en-token.bin", 1);
		
		VSM_Method.LoadStopwords("./data/stopwords.txt", analyzer.m_stopwords);
		VSM_Method.LoadWords("./data/features_Help.txt", analyzer.features);
		//VSM_Method.LoadWords("./data/corpus_Help.txt", analyzer.corpus_train);
		//System.out.println("Fuck Eclipse!");
		//analyzer.DF(analyzer.LoadJson("./data/Amazon/Amazon1.json"),analyzer.m_reviews);
		Methods.LoadDocID("./data/CrossValidationHelp/Train", ".txt", analyzer.corpus_train);
		Methods.LoadDocID("./data/CrossValidationHelp/Test", ".txt", analyzer.corpus_test);
		analyzer.EvaluatekNNClassifier("./data/Amazon", ".json");
		
		//analyzer.LoadDirectory("./data/Amazon","json");

		//System.out.println("Total # of docs in corpus:"+analyzer.corpus_train.size());
		//NaiveBayes.NBTrain(posdoc,negdoc,analyzer.features,analyzer.posiyes,analyzer.negyes);
		//analyzer.LoadTestDirectory("./data/query", ".json");
		//analyzer.calcFeature(analyzer.posiyes, analyzer.posino, analyzer.negyes, analyzer.negno);
		/*File file=new File("./data/corpus_PN.txt");
		PrintWriter writer=new PrintWriter(file);
		for(String s:analyzer.corpus_train) {
			writer.println(s);
		}
		writer.close();*/
	}
}
