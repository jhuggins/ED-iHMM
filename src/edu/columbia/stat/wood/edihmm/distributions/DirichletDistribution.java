/*
 * Created on Jun 29, 2011
 */
package edu.columbia.stat.wood.edihmm.distributions;

import cern.jet.stat.Gamma;

import edu.columbia.stat.wood.edihmm.util.Util;

/**
 * @author Jonathan Huggins
 *
 */
public class DirichletDistribution implements Distribution<Double[]> {

	private static final long serialVersionUID = 6598036467768225204L;

	// anything smaller than this is considered zero
	private static double MIN_NZ_PARAM = 1e-100; // 1e-5
	
	private double[] parameters;
	
	public DirichletDistribution(double[] parameters) {
		this.parameters = parameters;
	}

	public static double[] sample(double[] params) {
		double[] samp = new double[params.length];
		double sum = 0;
		for (int i = 0; i < samp.length; i++) {
			samp[i] = GammaDistribution.sample(params[i], 1);
			sum += samp[i];
		}
		for (int i = 0; i < samp.length; i++) {
			samp[i] /= sum;
		}
		return samp;
	}

	public Double[] sample() {
		return Util.boxArray(sample(parameters));  
	}


	public double logLikelihood(Double[] data) {
		return logLikelihood(parameters, Util.unboxArray(data));
	}
	
	public static double logLikelihood(double[] parameters, double[] data) {
		if (data.length != parameters.length) 
			throw new IllegalArgumentException("Dirichlet Distribution: dat and parameter length mismatch");
		double ll = 0;
		double psum = 0;
		for (int i = 0; i < data.length; i++) {
			if (parameters[i] > MIN_NZ_PARAM) {
				ll += (parameters[i] - 1)*Math.log(data[i]) - Gamma.logGamma(parameters[i]);
				psum += parameters[i];
			} else if (data[i] > MIN_NZ_PARAM) {
				return Double.NEGATIVE_INFINITY;
			}
		}
		ll += Gamma.logGamma(psum);
		
		return ll;
	}

	public double logPartition() {
		return logPartition(parameters);	
	}
	
	public static double logPartition(double[] params) {
		double l = 0;
		double sum = 0;
		for (double p : params) {
			if (p > MIN_NZ_PARAM) {
				l += Gamma.logGamma(p);
				sum += p;
			}
		}
		l -= Gamma.logGamma(sum);
		return l;
	}

	
}
