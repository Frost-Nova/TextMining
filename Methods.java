package TextClassifier;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.HashSet;

import VSM_LM.VSM_Method;

public class Methods{
	
	public static void partition(HashSet<String> corpus) {
		File file1 = new File("./data/1.txt");
		File file2 = new File("./data/2.txt");
		File file3 = new File("./data/3.txt");
		File file4 = new File("./data/4.txt");
		File file5 = new File("./data/5.txt");
		File file6 = new File("./data/6.txt");
		File file7 = new File("./data/7.txt");
		File file8 = new File("./data/8.txt");
		File file9 = new File("./data/9.txt");
		File file10 = new File("./data/10.txt");
		try {
			int i=0;
			PrintWriter writer1 = new PrintWriter(file1);
			PrintWriter writer2 = new PrintWriter(file2);
			PrintWriter writer3 = new PrintWriter(file3);
			PrintWriter writer4 = new PrintWriter(file4);
			PrintWriter writer5 = new PrintWriter(file5);
			PrintWriter writer6 = new PrintWriter(file6);
			PrintWriter writer7 = new PrintWriter(file7);
			PrintWriter writer8 = new PrintWriter(file8);
			PrintWriter writer9 = new PrintWriter(file9);
			PrintWriter writer10 = new PrintWriter(file10);
			for(String p:corpus) {
				if(i<3454)
					writer1.println(p);
				else if(i>=3454&&i<2*3454)
					writer2.println(p);
				else if(i>=2*3454&&i<3*3454)
					writer3.println(p);
				else if(i>=3*3454&&i<4*3454)
					writer4.println(p);
				else if(i>=4*3454&&i<5*3454)
					writer5.println(p);
				else if(i>=5*3454&&i<6*3454)
					writer6.println(p);
				else if(i>=6*3454&&i<7*3454)
					writer7.println(p);
				else if(i>=7*3454&&i<8*3454)
					writer8.println(p);
				else if(i>=8*3454&&i<9*3454)
					writer9.println(p);
				else if(i>=9*3454&&i<corpus.size())
					writer10.println(p);
				i++;
			}
			writer1.close();
			writer2.close();
			writer3.close();
			writer4.close();
			writer5.close();
			writer6.close();
			writer7.close();
			writer8.close();
			writer9.close();
			writer10.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}
	
	public static void LoadDocID(String folder, String suffix, HashSet<String> corpus) {
		File dir = new File(folder);
		for (File f : dir.listFiles()) {
			if (f.isFile() && f.getName().endsWith(suffix)) {
				VSM_Method.LoadWords(f.getAbsolutePath(),corpus);
			}
			else if (f.isDirectory())
				LoadDocID(f.getAbsolutePath(), suffix, corpus);
		}
		System.out.println("Loading " + corpus.size() + " review documents from " + folder);
	}
	
}