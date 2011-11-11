package edu.columbia.stat.wood.edihmm.distributions;

public class ParamTransitionProbs<P> {

	public double toProb;
	public double fromProb;
	public P param;
	
	/**
	 * Constructs a <tt>ParamTransitionProbs</tt> object.
	 *
	 * @param p
	 * @param param
	 */
	public ParamTransitionProbs(double toProb, double fromProb, P param) {
		this.toProb = toProb;
		this.fromProb = fromProb;
		this.param = param;
	}
	
}
