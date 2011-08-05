/*
 * Created on Jul 15, 2011
 */
package edu.columbia.stat.wood.edihmm.distributions;

/**
 * @author Jonathan Huggins
 *
 */
public class GaussianPair extends PriorDataDistributionPair<Double, Double> {

	private static final long serialVersionUID = -4745766477860588981L;
	
	double priorMean;
	double priorVar;
	double variance;
	
	/**
	 * 
	 * Constructs a <tt>GaussianPair</tt> object.
	 *
	 * @param priorMean
	 * @param variance
	 */
	public GaussianPair(double priorMean, double priorVar, double var) {
		super(new GaussianDistribution(priorMean,  priorVar));
		this.priorMean = priorMean;
		this.priorVar = priorVar;
		variance = var;
	}
	
	@Override
	public double dataLogLikelihood(Double mean, Double value) {
		double ll = GaussianDistribution.logLikelihood(mean,  variance, value);
		return ll;
	}

	@Override
	public double observationLogLikelihood(Iterable<Double> observations, Double param) {
		double sum = 0;
		double sumSq = 0;
		int n = 0;
		for (Double obs : observations) {
			sum += obs;
			sumSq += obs*obs;
			n++;
		}
		double postVar = 1/(1/priorVar + n/variance);
		double postMean = (priorMean/priorVar + sum/variance)*postVar;
		return (Math.log(postVar) - Math.log(priorVar) - n*Math.log(2*Math.PI*variance)
		        - sumSq/variance - priorMean/priorVar - postMean/postVar)/2;
	}

	@Override
	public Double sample(Double mean) {
		return GaussianDistribution.sample(mean,  Math.sqrt(variance));
	}

	@Override
	public Double samplePosterior(Iterable<Double> observations) {
		double sum = 0;
		double sumSq = 0;
		int n = 0;
		for (Double obs : observations) {
			sum += obs;
			sumSq += obs*obs;
			n++;
		}
		double postVar = 1/(1/priorVar + n/variance);
		double postMean = (priorMean/priorVar + sum/variance)*postVar;
		return GaussianDistribution.sample(postMean, Math.sqrt(postVar));
	}

}
