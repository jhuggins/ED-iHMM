/*
 * Created on Jun 30, 2011
 */
package edu.columbia.stat.wood.edihmm.distributions;

import java.io.Serializable;



/**
 * @author Jonathan Huggins
 *
 * @param P type of the the parameter for the data distribution
 * @param D type of the data distribution
 */
public abstract class PriorDataDistributionPair<P,D> implements Serializable {


	private static final long serialVersionUID = 6026259427131324953L;
	protected Distribution<P> priorDistr;
	
	/**
	 * 
	 * Constructs a <tt>PriorDataDistributionPair</tt> object.
	 *
	 * @param priorDistr
	 * @param dataDistr
	 */
	protected PriorDataDistributionPair(Distribution<P> priorDistr) {
		this.priorDistr = priorDistr;
	}
	
	/**
	 * Sample from the data distribution using the specified data distribution parameters
	 * 
	 * i.e., sample from p(data | param)
	 * 
	 * @param param
	 * @return
	 */
	public abstract D sample(P param);
	
	/**
	 * Sample from the posterior based on the set of observations, using
	 * at most only the first <tt>count</tt>
	 * 
	 * i.e., sample from p(param | observations)
	 * 
	 * @param observations collection of observations
	 * @param count number of observations to use
	 * @return posterior parameters for the data distribution 
	 */
	public abstract P samplePosterior(Iterable<D> observations);
	
	/**
	 * Sample from the prior distribution p(param)
	 * 
	 * @return
	 */
	public P samplePrior() {
		return priorDistr.sample();
	}
	
	/**
	 * log p(value | param)
	 * 
	 * @param param
	 * @param value
	 * @return
	 */
	public abstract double dataLogLikelihood(P param, D value);
	
	/**
	 * log p(observations | params)
	 * 
	 * @param observations
	 * @param count
	 * @return
	 */
	public abstract double observationLogLikelihood(Iterable<D> observations, P param);
	
}
