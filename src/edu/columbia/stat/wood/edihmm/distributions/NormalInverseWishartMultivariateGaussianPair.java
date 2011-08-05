/*
 * Created on Aug 4, 2011
 */
package edu.columbia.stat.wood.edihmm.distributions;

import edu.columbia.stat.wood.edihmm.util.Util;
import gov.sandia.cognition.math.matrix.*;
import gov.sandia.cognition.math.matrix.mtj.DenseMatrixFactoryMTJ;
import gov.sandia.cognition.math.matrix.mtj.DenseVectorFactoryMTJ;
import gov.sandia.cognition.statistics.distribution.InverseWishartDistribution;

/**
 * 
 * @author Jonathan Huggins
 *
 */
public class NormalInverseWishartMultivariateGaussianPair extends
		PriorDataDistributionPair<MultivariateGaussianParams, Vector> {

	private static final long serialVersionUID = 6418483115577228258L;
	
	Vector priorMean;
	double priorMeasurements;
	int df;
	Matrix scaleMatrix;
	InverseWishartDistribution iwd;
	gov.sandia.cognition.statistics.distribution.MultivariateGaussian g;
	
	public NormalInverseWishartMultivariateGaussianPair(Vector priorMean, double priorMeasurements, int df, Matrix scaleMatrix) {
		super(null);
		this.priorMean = priorMean;
		this.priorMeasurements = priorMeasurements;
		this.df = df;
		this.scaleMatrix = scaleMatrix;
		iwd = new InverseWishartDistribution(scaleMatrix, df);
	}
	
	@Override
	public double dataLogLikelihood(MultivariateGaussianParams param, Vector value) {
		return MultivariateGaussian.logLikelihood(param.mean, param.covariance, value);
	}

	@Override
	public double observationLogLikelihood(Iterable<Vector> observations, MultivariateGaussianParams params) {
		double ll = iwd.getProbabilityFunction().logEvaluate(params.covariance);
		ll += MultivariateGaussian.logLikelihood(priorMean, params.covariance.scale(1/priorMeasurements), params.mean);
		
		MultivariateGaussian g = new MultivariateGaussian(params.mean, params.covariance);
		for (Vector v : observations) {
			ll += g.logLikelihood(v);
		}
		
		return ll;
	}

	@Override
	public Vector sample(MultivariateGaussianParams param) {
		return MultivariateGaussian.sample(param.mean, param.covariance);
	}

	@Override
	public MultivariateGaussianParams samplePosterior(Iterable<Vector> observations) {
		Vector sum = DenseVectorFactoryMTJ.INSTANCE.createVector(priorMean.getDimensionality());
		int n = 0;
		for (Vector obs : observations) {
			sum.plusEquals(obs);
			n++;
		}
		double postMeasurements = n + priorMeasurements;
		int postDf = n + df;
		
		Vector obsMean = n > 0 ? sum.scale(1.0/n) : DenseVectorFactoryMTJ.INSTANCE.createVector2D();
		Vector postMean = priorMean.scale(priorMeasurements/postMeasurements).plus(obsMean.scale(n/postMeasurements));
		
		Vector meanMinusPriorMean = obsMean.minus(priorMean);
		Matrix postCov = meanMinusPriorMean.outerProduct(meanMinusPriorMean);
		postCov.scaleEquals(priorMeasurements*n/postMeasurements);
		
		Matrix sumSq = DenseMatrixFactoryMTJ.INSTANCE.createMatrix(scaleMatrix.getNumRows(), scaleMatrix.getNumColumns());
		n = 0;
		for (Vector obs : observations) {
			Vector obsMinusMean = obs.minus(obsMean);
			sumSq.plusEquals(obsMinusMean.outerProduct(obsMinusMean));
			n++;
		}
		Matrix postScale = scaleMatrix.plus(sumSq).plus(postCov);
		
		Matrix covariance = new InverseWishartDistribution(postScale, postDf).sample(Util.RAND);
		Vector mean = MultivariateGaussian.sample(postMean, covariance.scale(1./postMeasurements));
		return new MultivariateGaussianParams(mean, covariance);
	}
	
	@Override
	public MultivariateGaussianParams samplePrior() {
		Matrix covariance = iwd.sample(Util.RAND);
		Vector mean = MultivariateGaussian.sample(priorMean, covariance.scale(1./priorMeasurements));
		return new MultivariateGaussianParams(mean, covariance);
	}

}
