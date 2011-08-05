/*
 * Created on Aug 4, 2011
 */
package edu.columbia.stat.wood.edihmm.distributions;

import gov.sandia.cognition.math.matrix.Matrix;
import gov.sandia.cognition.math.matrix.Vector;

import java.io.Serializable;

/**
 * @author Jonathan Huggins
 *
 */
public class MultivariateGaussianParams implements Serializable {

	private static final long serialVersionUID = -8299873574416789930L;
	
	public Vector mean;
	public Matrix covariance;
	
	/**
	 * Constructs a <tt>MultivariateGaussianParams</tt> object.
	 *
	 * @param mean
	 * @param covariance
	 */
	public MultivariateGaussianParams(Vector mean, Matrix covariance) {
		this.mean = mean;
		this.covariance = covariance;
	}
	
	public String toString() {
		return "(mean = " + mean + ", covariance = " + covariance + ")"; 
	}
	
}
