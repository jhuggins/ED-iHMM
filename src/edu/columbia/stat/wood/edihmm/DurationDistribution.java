package edu.columbia.stat.wood.edihmm;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

import edu.columbia.stat.wood.edihmm.distributions.IntegerPriorDataDistributionPair;
import edu.columbia.stat.wood.edihmm.util.Range;

/**
 * A <tt>DurationDistribution</tt> object acts using some underlying 
 * <tt>IntegerPriorDataDistributionPair</tt> as its distribution.
 * It contains methods for keeping track of the duration parameters for 
 * each state. 
 * 
 * @author Jonathan Huggins
 *
 * @param P prior/parameter data type
 */
public class DurationDistribution<P> {

	private IntegerPriorDataDistributionPair<P> pddp;

	/**
	 * Auxiliary ArrayList used during updating; not
	 * re-instantiated for each update for efficiency
	 */
	private ArrayList<Integer> durations; 
	/**
	 * One set of parameters for each state
	 */
	private ArrayList<P> durationParams;
	
	/**
	 * 
	 * Constructs a <tt>EmissionDistribution</tt> object.
	 *
	 * @param pddp
	 */
	public DurationDistribution(IntegerPriorDataDistributionPair<P> pddp) {
		init(pddp);
	}
	
	/**
	 * 
	 * Constructs a <tt>EmissionDistribution</tt> object.
	 *
	 * @param pddp
	 * @param params
	 */
	public DurationDistribution(IntegerPriorDataDistributionPair<P> pddp, ArrayList<P> params) {
		init(pddp);
		durationParams = params;
	}
	

	public DurationDistribution(IntegerPriorDataDistributionPair<P> pddp, P[] params) {
		init(pddp);
		setParameters(params);
	}
	
	private void init(IntegerPriorDataDistributionPair<P> pddp) {
		this.pddp = pddp;
		durationParams = null;
		durations = null;
	}
	
	
	/**
	 * Duration ranges for each state duration distribution such
	 * that the probabilities outside this range are less than 
	 * <tt>minProb</tt>
	 * 
	 * @param minProb
	 */
	public Range[] durationRanges(double minProb) {
		if (minProb == 0) {
			throw new IllegalArgumentException("minProb must be greater than 0");
		}
		double minLL = Math.log(minProb);
		Range[] ranges = new Range[durationParams.size()];
//		System.out.print(minLL + " ");
		for(int i = 0; i < ranges.length; i++) {
//			System.out.print(durationParams.get(i) + " ");
			ranges[i] = pddp.range(durationParams.get(i), minLL);
		}
//		System.out.println(" * ");
		return ranges;
	}
	
	public double probability(int state, int duration) {
		return pddp.probability(durationParams.get(state), duration);
	}
	
	/**
	 * Sample parameters based on the state sequence information
	 * 
	 * @param states number of states
	 * @param ss state sequence
	 */
	public void update(int states, State[] ss) {
		if (durations == null) {
			durations = new ArrayList<Integer>(ss.length);
		}
		if (durationParams == null) {
			durationParams = new ArrayList<P>((int)(states*1.5)+1);
		}
		// update parameters for each state
		for (int k = 0; k < states; k++) {
			durations.clear();
			State prev = null;
			for (State s : ss) {
				if (s.getState() == k && (prev == null || prev.getDuration() == 0)) {
					durations.add(s.getDuration());
				}
				prev = s;
			}
			P samp = pddp.samplePosterior(durations);
			if (durationParams.size() == k) {
				durationParams.add(samp);
			} else {
				durationParams.set(k, samp);
			}
		}
	}
	
	/**
	 * Sample the duration for a given state
	 * 
	 * @param state
	 * @return a duration
	 */
	public int sample(int state) {
		assert 0 <= state && state < durationParams.size() : String.format("state = %d; valid states = %d", state, durationParams.size());
		return pddp.sample(durationParams.get(state));
	}
	
	/**
	 * Sample the duration distribution parameter from the prior
	 * 
	 * @return
	 */
	public P samplePrior() {
		return pddp.samplePrior();
	}
	
	/**
	 * Remove states that are no longer in use
	 * 
	 * @param states
	 */
	public void removeStates(List<Integer> states) {
		int offset = 0;
		for (int i : states) {
			durationParams.remove(i-offset);
			offset++;
		}
	}

	public void addState() {
		durationParams.add(samplePrior());;
	}
	
	public double logLikelihood(State[] ss) {
		ArrayList<LinkedList<Integer>> obs = new ArrayList<LinkedList<Integer>>(durationParams.size());
		for (int i = 0; i < durationParams.size(); i++) {
			obs.add(new LinkedList<Integer>());
		}
		State prevState = null;
		for (State s : ss) {
			if (prevState == null || prevState.getDuration() == 0) {
				obs.get(s.getState()).add(s.getDuration());
			}
			prevState = s;
		}
		
		double ll = 0;
		for (int i = 0; i < durationParams.size(); i++) {
			ll += pddp.observationLogLikelihood(obs.get(i), durationParams.get(i));
		}
		return ll;
	}
	
	public void printParams() {
		System.out.println("DD params = " + durationParams);
	}
	
	public int states() {
		return durationParams == null ? -1 : durationParams.size();
	}
	
	/**
	 * @return the pddp
	 */
	public IntegerPriorDataDistributionPair<P> getPddp() {
		return pddp;
	}
	
	@SuppressWarnings("unchecked")
	public P[] getParameters() {
		if (durationParams.size() == 0) {
			return (P[])new Object[0];
		}
		P[] array = (P[])Array.newInstance(durationParams.get(0).getClass(), durationParams.size());
		durationParams.toArray(array);
		return array;
	}
	
	public void setParameters(P[] params) {
		if (durationParams == null) {
			durationParams = new ArrayList<P>((int)(1.5*params.length));
		} else {
			durationParams.clear();
		}
		for (P p : params)
			durationParams.add(p);
	}
	
	@Override
	public String toString() {
		return String.format("DurationDistribution(pddp=%s, params=%s)", pddp, durationParams);
	}


}
