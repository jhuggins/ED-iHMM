/*
 * Created on Jul 1, 2011
 */
package edu.columbia.stat.wood.edihmm.distributions;

import java.util.Collection;

import cern.jet.stat.Gamma;


/**
 * @author Jonathan Huggins
 *
 */
public class GammaPoissonPair extends IntegerPriorDataDistributionPair<Double> {

	private static final long serialVersionUID = 4723000965485541676L;
	
	double shape; 
	double scale;
	
	public GammaPoissonPair(double shape, double scale) {
		super(new GammaDistribution(shape, scale));
		
		this.shape = shape;
		this.scale = scale;
		
	}

	@Override
	public double dataLogLikelihood(Double param, Integer value) {
		double ll = PoissonDistribution.likelihood(param, value);
		return ll;
	}

	@Override
	public Integer sample(Double param) {
		return PoissonDistribution.sample(param);
	}

	@Override
	public Double samplePosterior(Collection<Integer> observations) {
		int sum = 0;
		int n = 0;
		for (Integer obs : observations) {
			sum += obs;
			n++;
		}
		double s = GammaDistribution.sample(shape + sum, scale/(1 + n*scale));
		assert !Double.isInfinite(s) && !Double.isNaN(s);
		return s;
	}

	@Override
	public double observationLogLikelihood(Collection<Integer> observations, Double param) {
		int sum = 0;
		int n = 0;
		double ll = 0;
		for (Integer obs : observations) {
			ll -= Gamma.logGamma(obs + 1);
			sum += obs;
			n++;
		}
		ll += GammaDistribution.logPartition(shape + sum, scale/(1 + n*scale));
		ll -= GammaDistribution.logPartition(shape, scale);

		return ll;
	}

	@Override
	protected int mode(Double param) {
		return (int)(param+1)-1;
	}

	
}
