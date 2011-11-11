/*
 * Created on Jun 29, 2011
 */
package edu.columbia.stat.wood.edihmm;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

import edu.columbia.stat.wood.edihmm.distributions.PriorDataDistributionPair;

/**
 * The <tt>EmissionDistribution</tt> class contains the parameters for 
 * the posterior emissions distribution for each state, as well
 * as information on prior and data distributions themselves.
 * 
 * @author Jonathan Huggins
 *
 * @param P emission distribution parameter type
 * @param D emission distribution type
 */
public class EmissionDistribution<P, D> {

	private PriorDataDistributionPair<P, D> pddp;
	/**
	 * Auxiliary ArrayList used during updating; not
	 * re-instantiated each update for efficiency
	 */
	private ArrayList<ArrayList<D>> emissions; 
	/**
	 * One set of parameters for each state
	 */
	private ArrayList<P> emitParams;
	
	/**
	 * 
	 * Constructs a <tt>EmissionDistribution</tt> object.
	 *
	 * @param pddp
	 */
	public EmissionDistribution(PriorDataDistributionPair<P, D> pddp) {
		init(pddp);
	}
	
	/**
	 * 
	 * Constructs a <tt>EmissionDistribution</tt> object.
	 *
	 * @param pddp
	 * @param params
	 */
	public EmissionDistribution(PriorDataDistributionPair<P, D> pddp, ArrayList<P> params) {
		init(pddp);
		emitParams = params;
	}
	

	public EmissionDistribution(PriorDataDistributionPair<P, D> pddp, P[] params) {
		init(pddp);
		setParameters(params);
	}
	
	private void init(PriorDataDistributionPair<P, D> pddp) {
		this.pddp = pddp;
		emitParams = null;
		emissions = null;
	}
	
	public double logLikelihood(int state, D data) {
		return pddp.dataLogLikelihood(emitParams.get(state), data);
	}
	
	/**
	 * 
	 */
	public void update(int states, ArrayList<State[]> ss, ArrayList<D[]> data) {
		if (emissions == null) {
			emissions = new ArrayList<ArrayList<D>>(data.size()*2);
		}
		if (emitParams == null) {
			emitParams = new ArrayList<P>((int)(states*1.5));
		}
		// update parameters for each state
		int length = ss.size() > 0 ? ss.size() * ss.get(0).length : 10;
		for (int k = 0; k < states; k++) {
			if (emissions.size() == k) {
				emissions.add(new ArrayList<D>(length));
			} else {
				emissions.get(k).clear();
			}
		}

		int i = 0;
		for (State[] seq : ss) {
			D[] dat = data.get(i++);
			int j = 0;
			for (State s : seq) {
				emissions.get(s.getState()).add(dat[j++]);
			}
		}
		for (int k = 0; k < states; k++) {
			P samp = pddp.samplePosterior(emissions.get(k));
			if (emitParams.size() == k) {
				emitParams.add(samp);
			} else {
				emitParams.set(k, samp);
			}
		}
	}

	/**
	 * 
	 * @param state
	 * @return
	 */
	public D sample(int state) {
		assert 0 <= state && state < emitParams.size();
		//System.out.print(emitParams.get(state) + " ");
		return pddp.sample(emitParams.get(state));
	}
	
	/**
	 * Remove states that are no longer in use
	 * 
	 * @param states
	 */
	public void removeStates(List<Integer> states) {
		int offset = 0;
		for (int i : states) {
			emitParams.remove(i-offset);
			offset++;
		}
	}

	public void addState() {
		emitParams.add(pddp.samplePrior());
	}
	
	public double logLikelihood(ArrayList<State[]> ss, ArrayList<D[]> data) {
		ArrayList<LinkedList<D>> obs = new ArrayList<LinkedList<D>>(emitParams.size());
		for (int i = 0; i < emitParams.size(); i++) {
			obs.add(new LinkedList<D>());
		}
		for (int i = 0; i < ss.size(); i++) {
			State[] seq = ss.get(i);
			D[] dat = data.get(i);
			for (int j = 0; j < seq.length; j++) {
				obs.get(seq[j].getState()).add(dat[j]);
			}
		}

		double ll = 0;
		for (int i = 0; i < emitParams.size(); i++) {
			ll += pddp.observationLogLikelihood(obs.get(i), emitParams.get(i));
		}
		return ll;
	}
	
	public void printParams() {
		System.out.println("ED params = " + emitParams);
		
	}
	
	/**
	 * @return the pddp
	 */
	public PriorDataDistributionPair<P,D> getPddp() {
		return pddp;
	}
	
	public int states() {
		return emitParams == null ? -1 : emitParams.size();
	}
	
	@SuppressWarnings("unchecked")
	public P[] getParameters() {
		if (emitParams.size() == 0) {
			return (P[])new Object[0];
		}
		P[] array = (P[])Array.newInstance(emitParams.get(0).getClass(), emitParams.size());
		emitParams.toArray(array);
		return array;
	}
	
	public void setParameters(P[] params) {
		if (emitParams == null) {
			emitParams = new ArrayList<P>((int)(params.length * 1.5));
		} else {
			emitParams.clear();
		}
		for (P p : params)
			emitParams.add(p);
	}
	
	@Override
	public String toString() {
		return String.format("EmissionDistribution(pddp=%s, params=%s)", pddp, emitParams);
	}
	
}
