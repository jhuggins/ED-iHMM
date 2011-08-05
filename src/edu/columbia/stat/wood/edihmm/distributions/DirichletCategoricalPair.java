/*
 * Created on Jul 1, 2011
 */
package edu.columbia.stat.wood.edihmm.distributions;

import java.util.Arrays;

import edu.columbia.stat.wood.edihmm.util.Util;

/**
 * @author Jonathan Huggins
 *
 */
public class DirichletCategoricalPair extends PriorDataDistributionPair<Double[], Integer> {


	private static final long serialVersionUID = -7652902888943163938L;

	private double[] priorParams;
	
	public DirichletCategoricalPair(double[] priorParams) {
		super(new DirichletDistribution(priorParams));
		
		this.priorParams = priorParams;
	}

	@Override
	public double dataLogLikelihood(Double[] params, Integer value) {
		return CategoricalDistribution.logLikelihood(Util.unboxArray(params), value);
	}

	@Override
	public Integer sample(Double[] params) {
		return CategoricalDistribution.sample(Util.unboxArray(params));
	}

	@Override
	public Double[] samplePosterior(Iterable<Integer> observations) {
		double[] postParams = Arrays.copyOf(priorParams, priorParams.length);
		for (Integer obs : observations) {
			postParams[obs] += 1;
		}
		return Util.boxArray(DirichletDistribution.sample(postParams));
	}

	@Override
	public double observationLogLikelihood(Iterable<Integer> observations, Double[] params) {
		double[] counts = Arrays.copyOf(priorParams, priorParams.length);
		for (Integer obs : observations) {
			counts[obs]++;
		}
		
		return DirichletDistribution.logPartition(counts) - DirichletDistribution.logPartition(priorParams);
	}

	
}
