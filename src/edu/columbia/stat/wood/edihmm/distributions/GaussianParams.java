/*
 * Created on Jul 26, 2011
 */
package edu.columbia.stat.wood.edihmm.distributions;

import java.io.Serializable;

/**
 * @author Jonathan Huggins
 *
 */
public class GaussianParams implements Serializable {

	private static final long serialVersionUID = 7641157565785955403L;

	public double mean;
	public double variance;
	
	/**
	 * Constructs a <tt>GaussianParams</tt> object.
	 *
	 * @param mean
	 * @param variance
	 */
	public GaussianParams(double mean, double variance) {
		this.mean = mean;
		this.variance = variance;
	}
	
	public String toString() {
		return "(mean = " + mean + ", variance = " + variance + ")";
	}
	
}
