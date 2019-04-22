/**
 * 
 */
package structures;

import java.util.HashMap;
import java.util.Set;

import json.JSONArray;
import json.JSONException;
import json.JSONObject;

/**
 * @author hongning
 * @version 0.1
 * @category data structure
 * data structure for a Yelp review document
 * You can create some necessary data structure here to store the processed text content, e.g., bag-of-word representation
 */
public class Post {
	//unique review ID from Amazon
	String m_ID;		
	public void setID(String ID) {
		m_ID = ID;
	}
	
	public String getID() {
		return m_ID;
	}

	//author's displayed name
	String m_author;	
	public String getAuthor() {
		return m_author;
	}

	public void setAuthor(String author) {
		this.m_author = author;
	}
	
	//Summary
	String m_summary;
	public String getSummary() {
		return m_summary;
	}

	public void setSummary(String sum) {
		this.m_summary = sum;
	}

	//review text content
	String m_content;
	public String getContent() {
		return m_content;
	}

	public void setContent(String content) {
		if (!content.isEmpty())
			this.m_content = content;
	}
	
	public boolean isEmpty() {
		return m_content==null || m_content.isEmpty();
	}

	//Help measure [unhelpful,helpful] of the post
	JSONArray m_helpful;
	public JSONArray getHelp() {
		return m_helpful;
	}

	public void setHelp(JSONArray help) {
		this.m_helpful = help;
	}
	
	//overall rating to the business in this review
	double m_rating;
	public double getRating() {
		return m_rating;
	}

	public void setRating(double rating) {
		this.m_rating = rating;
	}

	short vote=0;
	public short getVote() {
		return vote;
	}
	
	public void setVote(short v) {
		this.vote=v;
	}
	
	public Post(String ID) {
		m_ID = ID;
	}
	
	int[] m_tokens;
	public int[] getHash() {
		return m_tokens;
	}
	
	public void setHash(int[] tokens) {
		m_tokens = tokens;
	}
	
	HashMap<String, Double> m_vector; // suggested sparse structure for storing the vector space representation with N-grams for this document
	public HashMap<String, Double> getVct() {
		return m_vector;
	}
	
	public void setVct(HashMap<String, Double> vct) {
		m_vector = vct;
	}
	
	public double similarity(Post p, Set<String> feature) {
		Double sim=0.0;
		Double l1=0.0;
		Double l2=0.0;
		for(String token: feature) {
			sim=sim+p.getVct().get(token)*this.getVct().get(token);
			l1=l1+Math.pow(p.getVct().get(token),2);
			l2=l2+Math.pow(this.getVct().get(token), 2);
		}
		l1=Math.sqrt(l1);
		l2=Math.sqrt(l2);
		sim=sim/(l1*l2);
		return sim;//compute the cosine similarity between this post and input p based on their vector space representation
	}
	
	public Post(JSONObject json) {
		try {
			m_ID = json.getString("reviewerID");
			//setAuthor(json.getString("reviewerName"));
			setHelp(json.getJSONArray("helpful"));
			setContent(json.getString("reviewText"));
			setRating(json.getDouble("overall"));
			setSummary(json.getString("summary"));
		} catch (JSONException e) {
			e.printStackTrace();
		}
	}
	
	public JSONObject getJSON() throws JSONException {
		JSONObject json = new JSONObject();
		
		json.put("reviewerID", m_ID);//must contain
		//json.put("reviewerName", m_author);//must contain
		json.put("helpful", m_helpful);//must contain
		json.put("reviewText", m_content);//must contain
		json.put("overall", m_rating);//must contain
		json.put("summary", m_summary);//must contain
		
		return json;
	}
}
