/*
 * Created on Aug 5, 2011
 */
package edu.columbia.stat.wood.edihmm.distributions;

import cern.jet.stat.Gamma;
import edu.columbia.stat.wood.edihmm.util.Util;
import gov.sandia.cognition.statistics.distribution.GeometricDistribution;

/**
 * @author Jonathan Huggins
 *
 */
public class BetaGeometricPair extends IntegerPriorDataDistributionPair<Double> {

	private static final long serialVersionUID = 4248972579356631104L;

	double a, b;
	
	public BetaGeometricPair(double a, double b) {
		super(new BetaDistribution(a, b));
		if (a <= 0 || b <= 0) {
			throw new IllegalArgumentException("parameters must be greater than zero");
		}
		this.a = a;
		this.b = b;
	}
	
	@Override
	protected int mode(Double param) {
		return 0;
	}

	@Override
	public double dataLogLikelihood(Double param, Integer value) {
		return (value-1)*Math.log(param) + Math.log(1- param);
	}

	@Override
	public double observationLogLikelihood(Iterable<Integer> observations, Double param) {
		int sum = 0;
		int n = 0;
		for (Integer obs : observations) {
			sum += obs;
			n++;
		}
		return Math.log(Gamma.beta(a + n, b + sum) / Gamma.beta(a, b));
	}

	@Override
	public Integer sample(Double param) {
		return new GeometricDistribution(1-param).sample(Util.RAND).intValue();
	}

	@Override
	public Double samplePosterior(Iterable<Integer> observations) {
		int sum = 0;
		int n = 0;
		for (Integer obs : observations) {
			sum += obs;
			n++;
		}
		return BetaDistribution.sample(a + n, b + sum);
	}

}
