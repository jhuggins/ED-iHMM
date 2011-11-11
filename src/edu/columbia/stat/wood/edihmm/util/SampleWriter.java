/*
 * Created on Aug 19, 2011
 */
package edu.columbia.stat.wood.edihmm.util;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.util.LinkedList;

import edu.columbia.stat.wood.edihmm.Sample;

/**
 * @author Jonathan Huggins
 *
 */
public class SampleWriter<P, E, D> implements IterationListener<P, E, D>  {

	private String outFile;
	private LinkedList<Sample<P, E, D>> samples;
	private int iterNumber;
	
	/**
	 * Constructs a <tt>SampleWriter</tt> object.
	 *
	 * @param outFile
	 */
	public SampleWriter(String outFile) {
		this.outFile = outFile;
		samples = new LinkedList<Sample<P, E, D>>();
		iterNumber = 0;
	}



	public void newSample(Sample<P, E, D>  s, boolean keep) {
		if (keep) {
			samples.add(s);
		}
		iterNumber++;
		ObjectOutputStream oos = null;
		try {
			oos = new ObjectOutputStream(new FileOutputStream(outFile));
			oos.writeInt(iterNumber);
			oos.writeObject(s);
			oos.writeObject(samples);
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			try {
				if (oos != null) {
					oos.close();
				}
			} catch (IOException e) {
			}
		}
		
	}
}
