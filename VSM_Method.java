package VSM_LM;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.HashSet;

import org.tartarus.snowball.SnowballStemmer;
import org.tartarus.snowball.ext.englishStemmer;

public class VSM_Method{
	
	public static void LoadStopwords(String filename, HashSet<String> stopwords) {
		try {
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
			String line;
			while ((line = reader.readLine()) != null) {
				//it is very important that you perform the same processing operation to the loaded stopwords
				//otherwise it won't be matched in the text content
				line = SnowballStemming(Normalization(line));
				if (!line.isEmpty())
					stopwords.add(line);
			}
			reader.close();
			System.out.format("Loading %d stopwords from %s\n", stopwords.size(), filename);
		} catch(IOException e){
			System.err.format("[Error]Failed to open file %s!!", filename);
		}
	}
	
	public static void LoadWords(String filename, HashSet<String> words) {
		try {
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
			String line;
			while ((line = reader.readLine()) != null) {
				if (!line.isEmpty())
					words.add(line);
			}
			reader.close();
			System.out.format("Loading %d docs from %s\n", words.size(), filename);
		} catch(IOException e){
			System.err.format("[Error]Failed to open file %s!!", filename);
		}
	}

	public static void calcIDF(int totdoc, String filename, HashMap<String, Double> stats) {
		try {
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
			String[] Input;
			String line;
			Double idf;
			while((line=reader.readLine())!=null) {
				Input=line.split("\\s+");
				String T=Input[0];
				idf=Math.log10(totdoc/Double.parseDouble(Input[1]));
				stats.put(T,idf);
			}
			reader.close();
			System.out.format("Loading %d features from %s\n", stats.size(), filename);
		} catch(IOException e){
			System.err.format("[Error]Failed to open file %s!!", filename);
		}
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
		// removing all English punctuation
		token = token.replaceAll("\\p{Punct}", ""); 
		token = token.toLowerCase(); 
		// recognize integers and doubles via regular expression and convert them to a special symbol "NUM"
		token = token.replaceAll("\\d+", "NUM");
		return token;
	}
}