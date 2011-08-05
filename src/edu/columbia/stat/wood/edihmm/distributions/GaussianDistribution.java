/*
 * Created on Jul 15, 2011
 */
package edu.columbia.stat.wood.edihmm.distributions;

import cern.jet.random.Normal;
import edu.columbia.stat.wood.edihmm.util.Util;

/**
 * @author Jonathan Huggins
 *
 */
public class GaussianDistribution implements Distribution<Double> {


	private static final long serialVersionUID = -8711841397871040906L;

	private double mean;
	private double var;
	
	public GaussianDistribution(double mean, double var) {
		this.mean = mean;
		this.var = var;
	}
	
	/* (non-Javadoc)
	 * @see edu.columbia.stat.wood.edihmm.distributions.Distribution#logLikelihood(java.lang.Object)
	 */
	public double logLikelihood(Double data) {
		return logLikelihood(mean, var, data);
	}
	
	public static double logLikelihood(double mean, double var, double data) {
		return -((data-mean)*(data-mean)/var + Math.log(2*Math.PI*var))/2;
	}

	/* (non-Javadoc)
	 * @see edu.columbia.stat.wood.edihmm.distributions.Distribution#logPartition()
	 */
	public double logPartition() {
		throw new RuntimeException("Method Not Implemented");
	}

	/* (non-Javadoc)
	 * @see edu.columbia.stat.wood.edihmm.distributions.Distribution#sample()
	 */
	public Double sample() {
		return sample(mean,  var);
	}
	
	public static Double sample(double mean, double variance) {
		return new Normal(mean,  Math.sqrt(variance), Util.RNG).nextDouble();
	}

}
