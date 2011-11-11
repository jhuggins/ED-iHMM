package edu.columbia.stat.wood.edihmm.distributions;

import java.util.Collection;

import cern.jet.stat.Gamma;
import edu.columbia.stat.wood.edihmm.util.Util;
import gov.sandia.cognition.statistics.distribution.GeometricDistribution;

public class UniformBetaOffsetGeometricPair extends
		IntegerPriorDataDistributionPair<OffsetGeometricParams> {

	private static final long serialVersionUID = 1L;
	private int max;
	private double a, b;
	
	public UniformBetaOffsetGeometricPair(int max, double a, double b) {
		super(null);
		this.max = max;
		this.a = a;
		this.b = b;
	}
	
	@Override
	protected int mode(OffsetGeometricParams param) {
		return param.offset;
	}

	@Override
	public double dataLogLikelihood(OffsetGeometricParams param, Integer value) {
		if (value < param.offset) 
			return Double.NEGATIVE_INFINITY;
		return (value-param.offset)*Math.log(param.p) + Math.log(1 - param.p); 
	}

	@Override
	public double observationLogLikelihood(Collection<Integer> observations, OffsetGeometricParams param) {
		int sum = 0;
		int n = 0;
		for (Integer obs : observations) {
			if (obs < param.offset)
				return Double.NEGATIVE_INFINITY;
			sum += obs - param.offset;
			n++;
		}
		return   Gamma.logGamma(a+n) + Gamma.logGamma(b + sum) - Gamma.logGamma(a + b + n + sum)
		       - Gamma.logGamma(a)   - Gamma.logGamma(b)       + Gamma.logGamma(a + b); 
	}

	@Override
	public Integer sample(OffsetGeometricParams param) {
		return param.offset + new GeometricDistribution(1-param.p).sample(Util.RAND).intValue();
	}

	@Override
	public OffsetGeometricParams samplePosterior(Collection<Integer> observations) {
		int sum = 0;
		int n = 0;
		int min = max;
		for (Integer obs : observations) {
			sum += obs;
			min = Math.min(min, obs);
			n++;
		}
		int offset = n > 0 ? min : Util.RAND.nextInt(min+1);
		double p = 1 - BetaDistribution.sample(a + n, b + sum - n*offset);
		return new OffsetGeometricParams(offset, p);
	}
	
	public OffsetGeometricParams samplePrior() {
		OffsetGeometricParams params = new OffsetGeometricParams(Util.RAND.nextInt(max+1), 1 - BetaDistribution.sample(a, b));
		return params;
	}
	
	public boolean sampleMH() {
		return true;
	}
	
	public ParamTransitionProbs<OffsetGeometricParams> updateMH(OffsetGeometricParams param, Collection<Integer> observations) {
		int sum = 0;
		int n = 0;
		int ub = max;
		for (Integer obs : observations) {
			sum += obs;
			ub = Math.min(ub, obs);
			n++;
		}
		int offset;
		if (param.offset == max) {
			offset = max - 1;
		} else if (param.offset >= ub && ub > 0) {
			offset = ub - 1;
		} else if (param.offset == 0) {
			if (ub > 0) {
				offset = 1;
			} else {
				offset = 0;
			}
		} else  {
			offset = param.offset + Math.max(0, Util.RAND.nextInt(2) * 2 - 1);
		}
	
		return new ParamTransitionProbs<OffsetGeometricParams>(transitionProb(param.offset, offset, ub), 
				transitionProb(offset, param.offset, ub),
				new OffsetGeometricParams(offset, 1 - BetaDistribution.sample(a + n, b + sum - n*offset)));
	}
	
	private double transitionProb(int oldv, int newv, int ub) {
		if (oldv == newv && newv == ub && ub == 0) {
			return 1;
		} else  if (oldv == max) {
			if (newv == max - 1)
				return 1;
			else
				return 0;
		} else if (oldv == ub) {
			if (newv == oldv && ub == 0) 
				return 1;
			else if (newv == ub + 1)
				return 1;
			else 
				return 0;
		} else if (Math.abs(oldv - newv) == 1) {
			return .5;
		}
		return 0;
	}

}
