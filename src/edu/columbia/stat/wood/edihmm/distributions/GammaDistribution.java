/*
 * Created on Jun 29, 2011
 */
package edu.columbia.stat.wood.edihmm.distributions;

import edu.columbia.stat.wood.edihmm.util.Util;


/**
 * @author Jonathan Huggins
 *
 */
public class GammaDistribution implements Distribution<Double> {
	
	private static final long serialVersionUID = 3112568089455429649L;
	
	private double shape;
	private double scale;
	
	
	/**
	 * Constructs a <tt>GammaDistribution</tt> object.
	 *
	 * @param shape
	 * @param scale
	 */
	public GammaDistribution(double shape, double scale) {
		this.shape = shape;
		this.scale = scale;
	}
	
	public static double sample(double shape, double scale) {
		if (shape == 0) {
			return 0.0;
		}
		double s =  new cern.jet.random.Gamma(shape, 1/scale, Util.RNG).nextDouble();
		return s;
	}

	public Double sample() {
		return sample(shape, scale);
	}

	public double logLikelihood(Double data) {
		return logLikelihood(shape, scale, data);
	}
	
	public static double logLikelihood(double shape, double scale, double data) {
		return (shape-1)*Math.log(data) - data/scale - cern.jet.stat.Gamma.logGamma(shape) - shape * Math.log(scale); 
	}

	public double logPartition() {
		return logPartition(shape, scale);
	}
	
	public static double logPartition(double shape, double scale) {
		return cern.jet.stat.Gamma.logGamma(shape) + shape * Math.log(scale);
	}
	
}
