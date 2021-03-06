/*
 * Created on Jun 30, 2011
 */
package edu.columbia.stat.wood.edihmm.util;

import edu.columbia.stat.wood.edihmm.Sample;

/**
 * Interface for a Listener that will be called after each iteration
 * of the ED-iHMM sampler.
 * 
 * @author Jonathan Huggins
 *
 */
public interface IterationListener<P,E,D> {

	/**
	 * 
	 * @param s
	 * @param keep
	 */
	public void newSample(Sample<P,E,D> s, boolean keep);
	
}
