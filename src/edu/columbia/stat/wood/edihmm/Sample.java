/*
 * Created on Jun 29, 2011
 */
package edu.columbia.stat.wood.edihmm;

import java.io.Serializable;

import edu.columbia.stat.wood.edihmm.distributions.IntegerPriorDataDistributionPair;
import edu.columbia.stat.wood.edihmm.distributions.PriorDataDistributionPair;


/**
 * @author Jonathan Huggins
 *
 * @param P emission distribution parameter type
 * @param E duration distribution parameter type
 * 
 **/
public class Sample<P, E, D> implements Serializable {

	private static final long serialVersionUID = -2362326026234651992L;
	
	/**
	 * Initial distribution
	 */
	public final double[] pi0;
	/**
	 * Transition matrix
	 */
	public final double[][] pi;
	/**
	 * Mixing proportions
	 */
	public final MixingProportions mp;
	/**
	 * Sample state sequence 
	 */
	public final State[] stateSequence;
	/**
	 * Joint log likelihood
	 */
	public final double jll;
	/**
	 * Hyperparameters
	 */
	public final Hyperparameters hypers;
	/**
	 * Distribution prior/data distributions pair
	 */
	public final IntegerPriorDataDistributionPair<D> durationDistribution;
	/**
	 * Duration parameters
	 */
	public final D[] durationParams;
	/**
	 * Emissions prior/data distributions pair
	 */
	public final PriorDataDistributionPair<P,E> emitDistribution;
	/**
	 * Emission parameters
	 */
	public final P[] emitParams; 
	/**
	 * 
	 */
	public final double[] predictiveProbabilities;
	
	/**
	 * Constructs a <tt>Sample</tt> object. Clones <tt>mp</tt> and makes 
	 * a copy of <tt>ss</tt>.
	 *
	 * @param pi
	 * @param mp
	 * @param stateSequence
	 * @param jll
	 * @param c0
	 * @param c1
	 */
	public Sample(double[] pi0, double[][] pi, MixingProportions mp, State[] ss, double jll, 
			Hyperparameters hypers, IntegerPriorDataDistributionPair<D> durationDistribution,
			D[] durationParams, PriorDataDistributionPair<P,E> emitDistribution, P[] emitParams,
			double[] predictiveProbabilities) {
		this.pi0 = pi0;
		this.pi = pi;
		this.mp = mp.clone();
		stateSequence = new State[ss.length];
		for (int i = 0; i < ss.length; i++) 
			stateSequence[i] = new State(ss[i]);
		this.jll = jll;
		this.hypers = hypers.clone();
		this.durationDistribution = durationDistribution;
		this.durationParams = durationParams;
		this.emitDistribution = emitDistribution;
		this.emitParams = emitParams;
		this.predictiveProbabilities = predictiveProbabilities;
	}
	
	
}
