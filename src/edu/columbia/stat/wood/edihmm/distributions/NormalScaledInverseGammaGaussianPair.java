/*
 * Created on Jul 25, 2011
 */
package edu.columbia.stat.wood.edihmm.distributions;

import java.io.Serializable;
import java.util.Collection;

import cern.jet.stat.Gamma;

/**
 * @author Jonathan Huggins
 *
 */
public class NormalScaledInverseGammaGaussianPair extends PriorDataDistributionPair<GaussianParams, Double> implements Serializable {

	private static final long serialVersionUID = -4259706130181875741L;
	
	double priorMean;
	double muScale;
	double varianceShape;
	double varianceScale;
	
	/**
	 * Constructs a <tt>NormalScaledInverseGammaGaussianPair</tt> object.
	 *
	 * @param priorDistr
	 * @param muPriorMean
	 * @param muScale
	 * @param varianceShape
	 * @param varianceScale
	 */
	public NormalScaledInverseGammaGaussianPair(double priorMean,
			double muScale, double varianceShape, double varianceScale) {
		super(null);
		this.priorMean = priorMean;
		this.muScale = muScale;
		this.varianceShape = varianceShape;
		this.varianceScale = varianceScale;
	}

	@Override
	public double dataLogLikelihood(GaussianParams params, Double value) {
		double ll = GaussianDistribution.logLikelihood(params.mean,  params.variance, value);
		return ll;
	}

	@Override
	public double observationLogLikelihood(Collection<Double> observations, GaussianParams param) {
		double sum = 0;
		double sumSq = 0;
		int n = 0;
		for (Double obs : observations) {
			sum += obs;
			sumSq += obs*obs;
			n++;
		}
		double obsMean = n == 0 ? 0 : sum/n;
		
		double postShape = varianceShape + n/2;
		double postScale = varianceScale + .5*sumSq - .5*n*obsMean*obsMean + n*muScale/(n + muScale) * (obsMean-priorMean)*(obsMean-priorMean);
		
		return (varianceShape == 0 || varianceScale == 0 ?  0
		     : varianceShape * Math.log(varianceScale) - Gamma.logGamma(varianceShape))
			 - postShape * Math.log(postScale) 		   + Gamma.logGamma(postShape)
			 + (muScale == 0 ? 0 : .5 * Math.log(muScale/(n+muScale))) 	  
			 - .5 * n * Math.log(2 * Math.PI);
	}

	@Override
	public Double sample(GaussianParams params) {
		return GaussianDistribution.sample(params.mean, params.variance);
	}

	@Override
	public GaussianParams samplePosterior(Collection<Double> observations) {
		double sum = 0;
		double sumSq = 0;
		int n = 0;
		for (Double obs : observations) {
			sum += obs;
			sumSq += obs*obs;
			n++;
		}
		double obsMean = n == 0 ? 0 : sum/n;
		
		double postShape = varianceShape + n/2.;
		double postScale = varianceScale + .5*sumSq - .5*n*obsMean*obsMean + n*muScale/(n + muScale) * (obsMean-priorMean)*(obsMean-priorMean);
		double variance = 1/GammaDistribution.sample(postShape, 1/postScale);
		
		double postMean = (priorMean*muScale+sum)/(muScale+n);
		double postVariance = variance/(muScale+n);
		double mean = GaussianDistribution.sample(postMean, postVariance);
		return new GaussianParams(mean, variance);
	}
	
	@Override
	public GaussianParams samplePrior() {
		double variance = 1/GammaDistribution.sample(varianceShape, varianceScale);
		double mean = GaussianDistribution.sample(priorMean, variance/muScale);
		return new GaussianParams(mean, variance);
	}

	
}
