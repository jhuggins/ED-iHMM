package edu.columbia.stat.wood.edihmm;

/**
 * Container for the transition matrix and initial state 
 * distribution.
 * 
 * @author Jonathan Huggins
 *
 */
public class TransitionProbabilities {

	public double[][] pi;
	public double[] pi0;
	
	/**
	 * Constructs a <tt>TransitionProbabilities</tt> object.
	 *
	 * @param pi
	 * @param pi0
	 */
	public TransitionProbabilities(double[][] pi, double[] pi0) {
		this.pi = pi;
		this.pi0 = pi0;
	}
	
}
