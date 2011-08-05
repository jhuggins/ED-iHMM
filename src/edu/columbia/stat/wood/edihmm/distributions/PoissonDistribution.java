/*
 * Created on Jul 1, 2011
 */
package edu.columbia.stat.wood.edihmm.distributions;

import cern.jet.stat.Gamma;

import cern.jet.random.Poisson;
import edu.columbia.stat.wood.edihmm.util.Util;


/**
 * @author Jonathan Huggins
 *
 */
public class PoissonDistribution implements Distribution<Integer> {

	private static final long serialVersionUID = 618604751838660800L;
	
	private double rate;
	
	/**
	 * Constructs a <tt>PoissonDistribution</tt> object.
	 *
	 * @param rate
	 */
	public PoissonDistribution(double rate) {
		this.rate = rate;
	}
	
	public static int sample(double rate) {
		return new Poisson(rate, Util.RNG).nextInt();
	}
	
	public static double likelihood(double rate, int data) {
		return (data * Math.log(rate) - rate - Gamma.logGamma(data + 1));
	}

	/* (non-Javadoc)
	 * @see edu.columbia.stat.wood.edihmm.util.Distribution#likelihood(java.lang.Object)
	 */
	public double logLikelihood(Integer data) {
		return likelihood(rate, data);
	}

	/* (non-Javadoc)
	 * @see edu.columbia.stat.wood.edihmm.util.Distribution#sample()
	 */
	public Integer sample()  {
		return sample(rate);
	}
	
	public double logPartition() {
		return logPartition(rate);
	}

	public static double logPartition(double rate) {
		return Math.exp(-rate);
	}

}
