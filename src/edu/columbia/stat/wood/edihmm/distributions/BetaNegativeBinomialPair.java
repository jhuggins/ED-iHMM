/*
 * Created on Aug 17, 2011
 */
package edu.columbia.stat.wood.edihmm.distributions;

import java.util.Collection;

import gov.sandia.cognition.statistics.distribution.NegativeBinomialDistribution;
import cern.jet.stat.Gamma;
import edu.columbia.stat.wood.edihmm.util.Util;

/**
 * @author Jonathan Huggins
 *
 */
public class BetaNegativeBinomialPair extends IntegerPriorDataDistributionPair<Double[]> {

	private static final double MAX_R = 1e5;
	
	private BetaDistribution beta;
	private GammaDistribution gamma;
	private static final NegativeBinomialDistribution.MaximumLikelihoodEstimator MLE = new NegativeBinomialDistribution.MaximumLikelihoodEstimator();
			
	public BetaNegativeBinomialPair(double a_p, double b_p, double a_r, double b_r) {
		super(null);
		beta = new BetaDistribution(a_p, b_p);
		gamma = new GammaDistribution(a_r, b_r);
	}

	private static final long serialVersionUID = -741620096866463761L;

	/* (non-Javadoc)
	 * @see edu.columbia.stat.wood.edihmm.distributions.IntegerPriorDataDistributionPair#mode(java.lang.Object)
	 */
	@Override
	protected int mode(Double[] param) {
		double r = param[0];
		double p = param[1];
		return Math.max(0, (int)(p*(r - 1)/(1 - p)));
	}

	/* (non-Javadoc)
	 * @see edu.columbia.stat.wood.edihmm.distributions.PriorDataDistributionPair#dataLogLikelihood(java.lang.Object, java.lang.Object)
	 */
	@Override
	public double dataLogLikelihood(Double[] param, Integer value) {
		double r = param[0];
		double p = param[1];
		try {
			return   Gamma.logGamma(r + value) -  Gamma.logGamma(r) - Gamma.logGamma(value + 1)
				   + r*Math.log(1 - p) + value*Math.log(p);
		} catch (ArithmeticException e) {
			throw e;
		}
	}

	/* (non-Javadoc)
	 * @see edu.columbia.stat.wood.edihmm.distributions.PriorDataDistributionPair#observationLogLikelihood(java.lang.Iterable, java.lang.Object)
	 */
	@Override
	public double observationLogLikelihood(Collection<Integer> observations, Double[] param) {
		double p = param[1];
		
		double ll = 0;
		for (Integer obs : observations) {
			ll += dataLogLikelihood(param, obs);
		}
		ll += beta.logLikelihood(p);
		
		return ll;
	}

	/* (non-Javadoc)
	 * @see edu.columbia.stat.wood.edihmm.distributions.PriorDataDistributionPair#sample(java.lang.Object)
	 */
	@Override
	public Integer sample(Double[] param) {
		return new NegativeBinomialDistribution(param[0], param[1]).sample(Util.RAND).intValue();
	}

	/* (non-Javadoc)
	 * @see edu.columbia.stat.wood.edihmm.distributions.PriorDataDistributionPair#samplePosterior(java.lang.Iterable)
	 */
	@Override
	public Double[] samplePosterior(Collection<Integer> observations) {
		double sum = 0;
		int n = 0;
		for (Integer obs : observations) {
			sum += obs;
			n++;
		}
		//double r = mean*mean/(var - mean);
		observations.add(1);
		double r = (n == 0) ? gamma.sample() : MLE.learn(observations).getR();
		if (Double.isNaN(r) || Double.isInfinite(r) || r > MAX_R) {
			r = gamma.sample();
		}
		double p = 1 - BetaDistribution.sample(beta.alpha + r*n, beta.beta + sum);
		
		return new Double[]{r, p};
	}
	
	@Override
	public Double[] samplePrior() {
		return new Double[]{ gamma.sample(), beta.sample() }; 
	}

}
