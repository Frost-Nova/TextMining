/**
 * 
 */
package structures;

import java.util.HashMap;

/**
 * @author hongning
 * Suggested structure for constructing N-gram language model
 */
public class LanguageModel {

	int m_N; // N-gram
	int m_V; // the vocabular size
	HashMap<String, Token> m_model; // sparse structure for storing the maximum likelihood estimation of LM with the seen N-grams
	LanguageModel m_reference; // pointer to the reference language model for smoothing purpose
	
	double m_lambda=0.1; // parameter for linear interpolation smoothing
	double m_delta; // parameter for absolute discount smoothing
	double m_sigma;
	public LanguageModel(int N, double delta,double sigma, HashMap<String, Token> stats, LanguageModel ref) {
		m_N = N;	m_delta=delta;	m_sigma=sigma;
		m_V = stats.size();
		m_model = stats;
		if(N>1)
			m_reference=ref;
		else
			m_reference=null;
	}
	
	public double calcTTMLProb(String token) {
		if(m_N>1)
			return (m_model.get(token).getValue()+m_delta)/m_reference.m_model.get(token.split("/")[0]).getValue();
		else
			return m_model.get(token).getValue()/87013;
	}

	public double calcWTMLProb(String token) {
		if(m_N>1)
			return (m_model.get(token).getValue()+m_sigma)/m_reference.m_model.get(token.split("/")[1]).getValue();
		else
			return m_model.get(token).getValue()/87013;
	}
	
	public double calcLinearSmoothedProb(String token) {
		if (m_N>1) 
			return (1.0-m_lambda) * calcTTMLProb(token) + m_lambda * m_reference.calcLinearSmoothedProb(token.split("-")[0]);
		else
			return (m_model.get(token).getValue()+m_delta)/(4959643+m_delta*m_V);
	}
	
	public double calcAbsoluteDiscountProb(String token) {
		return (m_model.get(token).getValue()-m_delta)/m_reference.m_model.get(token.split("-")[0]).getValue()
				+m_lambda* m_reference.calcLinearSmoothedProb(token.split("-")[0]);
	}
	
	//We have provided you a simple implementation based on unigram language model, please extend it to bigram (i.e., controlled by m_N)
	public String Tagsampling(String firsttag) {
		int Num = m_model.keySet().size();
		int Randix=(int)Math.floor(Math.random()*Num);
		String[] TT= new String[Num];int i=0;
		for(String token:m_model.keySet()) {
			TT[i]=token;i++;
		}
		if(!TT[Randix].split("/")[1].isEmpty()) {
			if (TT[Randix].split("/")[0].equals(firsttag))
				return TT[Randix];
		}else
			return "";
		return "";
	}
	
	public String Wordsampling(String tag) {
		int Num = m_model.keySet().size();
		int Randix=(int)Math.floor(Math.random()*Num);
		String[] WT=new String[Num]; int i=0;
		for(String token:m_model.keySet()) {
			WT[i]=token;i++;
		}
		if(!WT[Randix].split("/")[1].isEmpty()) {
			if (WT[Randix].split("/")[1].equals(tag))
				return WT[Randix];
		}else
				return "";
		return "";
	}
	
	//We have provided you a simple implementation based on unigram language model, please extend it to bigram (i.e., controlled by m_N)
	public double logLikelihood(Post review) {
		double likelihood = 0;
		for(String token:review.getTokens()) {
			if(m_model.containsKey(token))
				likelihood += Math.log(calcLinearSmoothedProb(token));
			else if(m_N>1){
				if(m_reference.m_model.containsKey(token.split("-")[0]))
					likelihood += Math.log(m_lambda * m_reference.calcLinearSmoothedProb(token.split("-")[0]));
			}
			else
				likelihood += Math.log(m_delta/(4959643+m_delta*m_V));
		}
		return likelihood;
	}
}
