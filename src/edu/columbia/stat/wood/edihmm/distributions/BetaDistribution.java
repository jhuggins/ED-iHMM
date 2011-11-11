/*
 * Created on Jul 11, 2011
 */
package edu.columbia.stat.wood.edihmm.distributions;


import edu.columbia.stat.wood.edihmm.util.Util;

/**
 * @author Jonathan Huggins
 *
 */
public class BetaDistribution implements Distribution<Double> {
	
	private static final long serialVersionUID = -900308524076606848L;
	
	double alpha;
	double beta;
	cern.jet.random.Beta betaDistr;
	
	public BetaDistribution(double a, double b) {
		if (a <= 0 || b <= 0) 
			throw new IllegalArgumentException("a = " + a + " and b = " + b + " must be greater than zero");
		betaDistr = new cern.jet.random.Beta(a, b, Util.RNG);
		alpha = a;
		beta = b;
	}
	
	
	public static double sample(double a, double b) {
		if (a == 0 && b == 0) {
			throw new IllegalArgumentException();
		}
		if (a == 0) {
			return 0;
		} 
		if (b == 0) {
			return 1;
		}
		
		return new cern.jet.random.Beta(a, b, Util.RNG).nextDouble();
	}
	
	public double logLikelihood(Double data) {
		return Math.log(betaDistr.pdf(data.doubleValue()));
	}
	
	public double probability(Double data) {
		return betaDistr.pdf(data.doubleValue());
	}
	

	public Double sample() {
		return betaDistr.nextDouble();
	}
	
	public void setParameters(double a, double b) {
		betaDistr.setState(a, b);
		alpha = a;
		beta = b;
	}


	public double logPartition()  {
		throw new RuntimeException("Method Not Implemented");
		//return Gamma.logBeta(alpha, scale);
	}

}
