package edu.columbia.stat.wood.edihmm.distributions;

import edu.columbia.stat.wood.edihmm.util.Util;
import gov.sandia.cognition.statistics.distribution.BinomialDistribution;

import java.util.Collection;

import cern.jet.stat.Gamma;

public class BetaBinomialPair extends IntegerPriorDataDistributionPair<Double> {

	private static final long serialVersionUID = 1960290357029018288L;

	private int n;
	private double a, b;
	
	public BetaBinomialPair(int n, double a, double b) {
		super(new BetaDistribution(a,b));
		this.a = a;
		this.b = b;
		this.n = n;
	}
	
	@Override
	protected int mode(Double param) {
		return (int)((n+1)*param);
	}

	@Override
	public double dataLogLikelihood(Double p, Integer k) {
		if (k < 0 || k > n) 
			return Double.NEGATIVE_INFINITY;
		return Gamma.logGamma(n + 1) - Gamma.logGamma(k + 1) - Gamma.logGamma(n - k + 1) 
		+ k * Math.log(p) + (n - k) * Math.log(1 - p);
	}

	@Override
	public double observationLogLikelihood(Collection<Integer> observations,
			Double p) {
		
		double ll = 0;
		for (Integer obs : observations) {
			ll += dataLogLikelihood(p, obs);
		}
		
		ll += this.priorDistr.logLikelihood(p);
		
		return ll;
	}

	@Override
	public Integer sample(Double p) {
		return new BinomialDistribution(n,p).sample(Util.RAND).intValue();
	}

	@Override
	public Double samplePosterior(Collection<Integer> observations) {
		double sum = 0;
		for (Integer obs : observations) {
			sum += obs;
		}
		
		return BetaDistribution.sample(a + sum, b + n*observations.size() - sum);
	}

	
	
}
