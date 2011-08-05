package edu.columbia.stat.wood.edihmm.distributions;

import java.io.Serializable;

public interface Distribution<E> extends Serializable {

	/**
	 * 
	 * @return
	 */
	public E sample();
	
	/**
	 * 
	 * @param data
	 * @return
	 */
	public double logLikelihood(E data);
	
	/**
	 * The log of the partition function
	 * 
	 * @return
	 */
	public double logPartition();
	
}
