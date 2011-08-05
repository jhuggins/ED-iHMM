/*
 * Created on Jun 30, 2011
 */
package edu.columbia.stat.wood.edihmm.distributions;

import java.util.Arrays;

import edu.columbia.stat.wood.edihmm.util.Util;



/**
 * @author Jonathan Huggins
 *
 */
public class CategoricalDistribution implements Distribution<Integer> {

	private static final long serialVersionUID = 2962580177377182303L;
	private double[] parameters;
	
	/**
	 * Constructs a <tt>CategoricalDistribution</tt> object.
	 *
	 * @param parameters
	 */
	public CategoricalDistribution(double[] parameters) {
		this.parameters = parameters;
	}

	/**
	 * 
	 * @param parameters
	 * @return
	 * @throws IllegalArgumentException
	 */
	public static int sample(double[] parameters) throws IllegalArgumentException {
		if (Math.abs(Util.sum(parameters)-1.0) > 1e-5) {
			throw new IllegalArgumentException("CategoricalDistribution: parameters must sum to 1");
		}
		double rnd = Util.RNG.nextDouble();
		double sum = 0;
		for (int i = 0; i < parameters.length; i++) {
			sum += parameters[i];
			if (rnd < sum) {
				return i;
			}
		}
		// should never get here
		throw new RuntimeException("Categorical Distribution: sampling failed to return a value: " + rnd + " "+ Arrays.toString(parameters));
	}

	/**
	 * 
	 */
	public Integer sample() {
		return CategoricalDistribution.sample(parameters);
	}
	
	/**
	 * 
	 * @param parameters
	 * @param data
	 * @return
	 */
	public static double logLikelihood(double[] parameters, Integer data) {
		return Math.log(parameters[data]);
	}

	/**
	 * 
	 */
	public double logLikelihood(Integer data) {
		return Math.log(parameters[data]);
	}

	public double logPartition() {
		return 0;
	}

	public static double logPartition(double[] params) {
		return 0;
	}
	
}
