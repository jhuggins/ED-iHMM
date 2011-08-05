/*
 * Created on Jul 11, 2011
 */
package edu.columbia.stat.wood.edihmm.util;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;

/**
 * @author Jonathan Huggins
 *
 */
public class CharTextDocument {

	public Integer[] data;
	public int vocabSize;
	public char[] map;
	public HashMap<Character, Integer> inverseMap;
	
	public CharTextDocument(File file) throws FileNotFoundException {
		this(file, Integer.MAX_VALUE);
	}
	
	public CharTextDocument(File file, int maxLength) throws FileNotFoundException {
		Scanner in = new Scanner(file);
		StringBuilder sb = new StringBuilder();
		while(in.hasNext() && sb.length() < maxLength) {
			sb.append(in.nextLine() + "\n");
		}
		if (sb.length() > maxLength)
			init(sb.substring(0, maxLength));
		else 
			init(sb.toString());
	}
	
	public CharTextDocument(String text) {
		init(text);
	}
	
	private void init(String text ) {
		data = new Integer[text.length()];
		inverseMap = new HashMap<Character, Integer>();
		vocabSize = 0;
		int i = 0;
		for (char c : text.toCharArray()) {
			if (!inverseMap.containsKey(c)) {
				inverseMap.put(c, vocabSize);
				vocabSize++;
			}
			data[i] = inverseMap.get(c);
			i++;
		}
		map = new char[vocabSize];
		for (Map.Entry<Character, Integer> e : inverseMap.entrySet()) {
			map[e.getValue()] = e.getKey();
		}
	}
	
}
