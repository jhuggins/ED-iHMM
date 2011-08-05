/*
 * Created on Jun 30, 2011
 */
package edu.columbia.stat.wood.edihmm;

import java.io.Serializable;

/**
 * Hyperparameters for the ED-iHMM
 * 
 * @author Jonathan Huggins
 *
 */
public class Hyperparameters implements Serializable {

	private static final long serialVersionUID = 6384440342606957889L;
	
	/**
	 * Parameter for the geometric distribution 
	 * associated with the augmenting space of the 
	 * underlying Gamma Process. 
	 */
	public double q;
	/**
	 * Top level concentration parameter
	 */
	public double c0;
	/**
	 * Lower level concentration parameter
	 */
	public double c1;
	/**
	 * c0 ~ Gamma(c0_a, c0_b)
	 */
	public double c0_a;
	/**
	 * c0 ~ Gamma(c0_a, c0_b)
	 */
	public double c0_b;
	/**
	 * c1 ~ Gamma(c1_a, c1_b)
	 */
	public double c1_a;
	/**
	 * c1 ~ Gamma(c1_a, c1_b)
	 */
	public double c1_b;
	public double a_v;
	public double b_v;
	/**
	 * Temperature parameter K parameterizing the scaled Beta 
	 * distribution from which the auxiliary variables (u_t's)
	 * are sampled:
	 * <pre>    u_t/p(z_t|z_t-1) ~ Beta(1/K, K)</pre>
	 */
	public double temperature;
	
	/**
	 * 
	 * Constructs a <tt>Hyperparameters</tt> object.
	 *
	 */
	public Hyperparameters() {
		this(.9, 1, 1, 4, 2, 3, 5, 6, 1, 1);
	}
	/**
	 * Constructs a <tt>Hyperparameters</tt> object.
	 *
	 * @param c0
	 * @param c1
	 * @param c0_a
	 * @param c0_b
	 * @param c1_a
	 * @param c1_b
	 * @param a_v
	 * @param b_v
	 */
	public Hyperparameters(double q, double c0, double c1, double c0A, double c0B,
			double c1A, double c1B, double av, double bv, double temp) {
		this.q = q;
		this.c0 = c0;
		this.c1 = c1;
		c0_a = c0A;
		c0_b = c0B;
		c1_a = c1A;
		c1_b = c1B;
		a_v = av;
		b_v = bv;
		temperature = temp;
	}
	
	@Override
	public Hyperparameters clone() {
		return new Hyperparameters(q, c0, c1, c0_a, c0_b, c1_a, c1_b, a_v, b_v, temperature);
	}
}
