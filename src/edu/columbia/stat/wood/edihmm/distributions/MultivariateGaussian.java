/*
 * Created on Aug 4, 2011
 */
package edu.columbia.stat.wood.edihmm.distributions;

import edu.columbia.stat.wood.edihmm.util.Util;
import gov.sandia.cognition.math.matrix.Matrix;
import gov.sandia.cognition.math.matrix.Vector;


/**
 * @author Jonathan Huggins
 *
 */
public class MultivariateGaussian implements Distribution<Vector> {

	private static final long serialVersionUID = -8223989192441463769L;
	
	gov.sandia.cognition.statistics.distribution.MultivariateGaussian gaussian;
	
	public MultivariateGaussian(Vector mean, Matrix covariance) {
		gaussian = new gov.sandia.cognition.statistics.distribution.MultivariateGaussian(mean, covariance);
	}
	
	public double logLikelihood(Vector data) {
		return gaussian.getProbabilityFunction().logEvaluate(data);
	}
	

	public double logPartition() {
		throw new RuntimeException("Not implemented");
	}

	public Vector sample() {
		return gaussian.sample(Util.RAND);
	}
	
	public static double logLikelihood(Vector mean, Matrix covariance, Vector data) {
		return new gov.sandia.cognition.statistics.distribution.MultivariateGaussian.PDF(mean, covariance).logEvaluate(data);
	}
	

	public static Vector sample(Vector mean, Matrix covariance) {
		return gov.sandia.cognition.statistics.distribution.MultivariateGaussian.sample(mean, covariance, Util.RAND);
	}

}
