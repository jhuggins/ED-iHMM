/*
 * Created on Jun 29, 2011
 */
package edu.columbia.stat.wood.edihmm;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

import cern.colt.Timer;
import cern.jet.random.Normal;

import edu.columbia.stat.wood.edihmm.MixingProportions.Mode;
import edu.columbia.stat.wood.edihmm.distributions.BetaDistribution;
import edu.columbia.stat.wood.edihmm.distributions.CategoricalDistribution;
import edu.columbia.stat.wood.edihmm.distributions.DirichletDistribution;
import edu.columbia.stat.wood.edihmm.distributions.GammaDistribution;
import edu.columbia.stat.wood.edihmm.util.IterationListener;
import edu.columbia.stat.wood.edihmm.util.Range;
import edu.columbia.stat.wood.edihmm.util.Util;

/**
 * The explicit duration infinite HMM (ED-iHMM), parameterized by the types of 
 * the emissions distribution (<tt>E</tt>), the emissions distribution's 
 * parameter (<tt>P</tt>), and the duration distribution (<tt>D</tt>)
 * 
 * @author Jonathan Huggins
 *
 * @param P emission distribution parameter type
 * @param E emission distribution type
 * @param D duration distribution parameter type
 */
public class EDiHMM<P,E,D> {

	private static final double AUX_MAX_START = 1e-4;
//	private static final int SPLIT_MERGE_INTERVAL = 5;
	private static final int MAX_BREAKS = 20;
	private static final int MH_HYPER_ITERS = 10;
	public static final State START_STATE = new State(0,-1);

	private Hyperparameters hypers;
	private BetaDistribution auxDistr;
	private int numStates;
	private EmissionDistribution<P,E> ed;
	private DurationDistribution<D> dd;
	private MixingProportions mp;
	/**
	 * Transition matrix
	 */
	private double[][] pi;
	/**
	 * Initial state distribution
	 */
	private double[] pi0;
	/**
	 * Auxiliary r.v.'s
	 */
	private double[] auxArray;
	/**
	 * The state sequence
	 */
	private State[] ss;
	/**
	 * The data
	 */
	private E[] data;
	/**
	 * joint loglikelihood for each iteration
	 */
	private double[] jll;

	/**
	 * Listeners
	 */
	private LinkedList<IterationListener> iterListeners;


	/**
	 * Constructs a <tt>EDiHMM</tt> object from a sample. 
	 *
	 * @param sample
	 */
	public EDiHMM(Sample<P,E,D> sample) {
		this(new EmissionDistribution<P,E>(sample.emitDistribution, sample.emitParams), 
				new DurationDistribution<D>(sample.durationDistribution, sample.durationParams),
				sample.pi, sample.pi0);
		mp = sample.mp;
		ss = sample.stateSequence;
		hypers = sample.hypers;
		
		iterListeners = new LinkedList<IterationListener>();
	}

	/**
	 * 
	 * Constructs a <tt>EDiHMM</tt> object.
	 *
	 * @param ed
	 * @param dd
	 * @param pi
	 * @param pi0
	 */
	public EDiHMM(EmissionDistribution<P,E> ed, DurationDistribution<D> dd, double[][] pi, double[] pi0) {
		this.ed = ed;
		this.dd = dd;
		mp = null;
		this.pi = pi;
		this.pi0 = pi0;
		auxArray = null;
		data = null;
		ss = null;
		hypers = null;
		numStates = pi == null ? 0 : pi.length;

		iterListeners = new LinkedList<IterationListener>();
	}

	/**
	 * 
	 * Constructs a <tt>EDiHMM</tt> object.
	 *
	 * @param ed
	 * @param dd
	 */
	public EDiHMM(EmissionDistribution<P,E> ed, DurationDistribution<D> dd) {
		this(ed, dd, null, null);
	}

	/**
	 * Fill the array with generated data and return the log-likelihood 
	 * of the data. 
	 * 
	 * @return 
	 */
	public double generateData(E[] genData, Hyperparameters hypers) {
		this.hypers = hypers;
		ss = sampleStateSequence(genData.length, numStates);
		for (int i = 0; i < genData.length; i++) {
			genData[i] = ed.sample(ss[i].getState());
		}
		data = genData;
		mp =  new MixingProportions(1, new int[numStates][numStates+1], hypers.q, MixingProportions.Mode.GEOM);
		int[][] obsCounts = calculateTransitionCounts();
		int[][] rCounts = sampleRCounts(obsCounts);
		mp.sampleParameters(hypers.c0, rCounts);
		sampleHyperparameters(obsCounts);
		rCounts = sampleRCounts(obsCounts);
		mp.sampleParameters(hypers.c0, rCounts);
		
		return jll(obsCounts);
	}

	/**
	 * Get the current state sequence
	 * 
	 * @return the ss
	 */
	public State[] getStateSequence() {
		return ss;
	}

	/**
	 * 
	 * Set the state sequence.
	 * 
	 * @param ss the new state sequence
	 */
	public void setStateSequence(State[] ss) {
		this.ss = ss;
	}

	public static <P,E,D> int[] calculateStateCountDistribution(List<Sample<P,E,D>> samples) {
		int max = 0;
		for (Sample<P,E,D> s : samples) 
			max = Math.max(max, s.pi.length);

		int[] counts = new int[max+1];
		for (Sample<P,E,D> s : samples) 
			counts[s.pi.length]++;

		return counts;
	}

	public static <P,E> double predictivePerplexity(double[] probabilities) {
		double logsum = 0;
		for (double p : probabilities) {
			logsum += Math.log(p); 
		}
		return Math.pow(2, -logsum/probabilities.length/Math.log(2));
	}

	public static <P,E,D> double predictivePerplexity(Sample<P,E,D> sample) {
		return predictivePerplexity(sample.predictiveProbabilities);
	}

	/**
	 * 
	 */
	public static <P,E,D> double predictivePerplexity(List<Sample<P,E,D>> samples) {
		if (samples.isEmpty()) {
			return 0;
		}
		double[] probSums = new double[samples.get(0).predictiveProbabilities.length];
		for (Sample<P,E,D> s : samples) {
			for (int i = 0; i < probSums.length; i++) {
				probSums[i] += s.predictiveProbabilities[i];
			}
		}
		double logsum = 0;
		for (int i = 0; i < probSums.length; i++) {
			logsum += Math.log(probSums[i]/samples.size());
		}
		
		return Math.pow(2, -logsum/probSums.length/Math.log(2));
	}

	public double predictivePerplexity() {
		return predictivePerplexity(predictiveProbabilities());
	}

	public double[] predictiveProbabilities() {
		double[] probs = new double[data.length];
		double prevJLL = 0;
		for (int i = 0; i < probs.length; i++) {
			E[] tempData = Arrays.copyOf(data, i+1);
			State[] tempSS = Arrays.copyOf(ss, i+1);
			int[][] obsCounts = calculateTransitionCounts(tempSS);
			// emissions and duration LLs
			double jll = ed.logLikelihood(tempSS, tempData);
			jll += dd.logLikelihood(tempSS);
			
			// transition LL
			for (int m = 0; m < obsCounts.length; m++) {
				double[] c1b = mp.getBetas(m);
				Util.multiply(hypers.c1, c1b);
				jll += DirichletDistribution.logPartition(Util.addArrays(c1b, obsCounts[m])) - DirichletDistribution.logPartition(c1b);
			}
			probs[i] = Math.exp(jll - prevJLL);
			prevJLL = jll;
		}
		
		return probs;
	}

	/**
	 * @return the jll
	 */
	public double[] getJLL() {
		return jll;
	}

	/**
	 * Sample from the ED-iHMM, with the initial parameters sampled from the prior. 
	 * 
	 *  @see edu.columbia.stat.wood.edihmm.EDiHMM#sample(E[], Hyperparameters, int, int, int, int, State[])
	 * @param data
	 * @param hypers
	 * @param burnin
	 * @param interval
	 * @param samples
	 * @param states
	 * @return
	 */
	public List<Sample<P,E,D>> sample(E[] data, Hyperparameters hypers, int burnin, int interval, int samples, int states){
		return sample(data, hypers, burnin, interval, samples, states, null);
	}

	/**
	 * Sample from the ED-iHMM, with the initial parameters sampled from the prior if <tt>ss == null</tt>
	 * 
	 * @param data
	 * @param hypers
	 * @param burnin
	 * @param interval
	 * @param samples
	 * @param states
	 * @param ss
	 * 
	 * @return
	 */
	public List<Sample<P,E,D>> sample(E[] data, Hyperparameters hypers, int burnin, int interval, int samples, int states, State[] ss) {
		if (ss != null && data.length != ss.length) {
			throw new IllegalArgumentException("data and state sequence length are not equal");
		} else if (burnin < 0 || interval < 1 || samples < 1 || states < 1) {
			throw new IllegalArgumentException("invalid sampling parameter");
		}

		this.hypers = hypers;
		this.auxDistr = new BetaDistribution(1/hypers.temperature, hypers.temperature);
		this.data = data;
		this.numStates = states;
		this.ss = ss == null ? new State[data.length] : ss;

		auxArray = new double[data.length];

		// initial mixing proportions and transition matrix
		int[][] obsCounts = ss != null ? calculateTransitionCounts() : new int[states][states+1];
		if (hypers.q > 0) {
			mp = new MixingProportions(hypers.c0, obsCounts, hypers.q, Mode.GEOM);
		} else {
			mp = new MixingProportions(hypers.c0, obsCounts, hypers.a_v, hypers.b_v, Mode.GEOM);
		}
		TransitionProbabilities tp;
		if (ss != null) {
			tp = mp.sampleTransitionMatrix(hypers.c1, obsCounts, ss[0].getState());
		} else {
			tp = mp.sampleTransitionMatrix(hypers.c1, obsCounts); 
		}
		pi = tp.pi;
		pi0 = tp.pi0;

		// sample duration and emission parameters from their priors if necessary
		if (dd.states() != states) {
			dd.update(states, new State[] {});
		}
		if (ed.states() != states) {
			ed.update(states, new State[] {}, (E[])new Object[] {}); 
		}

		// generate a starting state sequence, if necessary
//		if (ss == null) {
//			this.ss = sampleStateSequence(data.length, states);
//			tp = mp.sampleTransitionMatrix(hypers.c1, calculateTransitionCounts(), this.ss[0].getState());  
//			pi = tp.pi;
//			pi0 = tp.pi0;
//		} else{
//			this.ss = ss;
//		}

		System.out.println(mp);

		// sample
		jll = new double[burnin + interval*(samples-1) + 1];
		LinkedList<Sample<P,E,D>> sampList = new LinkedList<Sample<P,E,D>>();

		Sample<P,E,D> sample;
		for (int i = 0; i < burnin; i++) {
			while ((sample = sample(i+1, false)) == null)
				continue;
			passSampleToListeners(sample, false);
		}
		for (int i = 0; i <= interval*(samples-1); i++) {
			boolean keep = i % interval == 0;
			while ((sample = sample(i+burnin+1, keep)) == null)
				continue;
			passSampleToListeners(sample, keep);
			if (keep) {
				sampList.add(sample);
			}
		}
//		System.out.println("split-merge accept proportion = " + ((double)accepts)/total);

		return sampList;
	}

	/**
	 * Sample everything once, and return the sample if sampling is successful . Otherwise return null.
	 * 
	 * @param n
	 * @param keep
	 * @return the sample if sampling is successful and null otherwise
	 */
	private Sample<P,E,D> sample(int n, boolean keep) {
		
		double minAux = AUX_MAX_START;
		//		System.out.println("n = " + n);
		if (n > 1) {
			minAux = sampleAuxiliaryVars(); 
		} else {
			Arrays.fill(auxArray, minAux);
		}
//		System.out.println("1");
		extendStateSpace(minAux);
//		System.out.println("2");
		ArrayList<LinkedHashMap<State,ArrayList<State>>> reachableStates = reachableStates(minAux);
//		System.out.println("3");
		if (reachableStates == null) {
			System.err.println("No path through state space. Resampling...");
			sampleParameters(calculateTransitionCounts());
			return null;
		}

		ArrayList<HashMap<State, Double>> alphas = forwardPass(reachableStates); 
		
		if (alphas == null) {
			System.err.println("No path through state space. Resampling...");
			sampleParameters(calculateTransitionCounts());
			return null;
		}
//		System.out.println("4");
		backwardSample(alphas);
//		System.out.println("5");
		cleanupStateSpace();
		
//		System.out.println("6");
		int[][] obsCounts = calculateTransitionCounts();
		
		sampleParameters(obsCounts);
//		System.out.println("7");
//		if (n > 29 && n % SPLIT_MERGE_INTERVAL == 0) {
//			//splitMerge();
//			restrictedForwardBackward();
//			System.out.println("split-merge accept proportion = " + ((double)accepts)/total);
//			obsCounts = calculateTransitionCounts();
//		}
//		System.out.println("8");
		jll[n-1] = jll(obsCounts, false);
//		System.out.println("9");
		System.out.printf("%d jll=%.4f c0=%.4f c1=%.4f q=%.4f states=%d norm_gamma=%s\n", n, jll[n-1], hypers.c0, hypers.c1, mp.q, numStates, Arrays.toString(mp.getNormalizedGammas()));
		ed.printParams();
		dd.printParams();
		System.out.println(mp);

		return new Sample<P,E,D>(pi0, pi, mp, ss, jll[n-1], hypers, dd.getPddp(), dd.getParameters(), ed.getPddp(), ed.getParameters(), keep ? predictiveProbabilities() : null) ;
	}

	private void extendStateSpace(double minAux) {
		int i = 0;
		while(Util.columnMax(pi, numStates) > minAux) {
			//			System.out.print(i + " ");
			if (i++ == MAX_BREAKS) {
				System.err.println("Exceeded maximum number of extra stick breaks. Unexpected behavior may occur.");
				break;
			}
			TransitionProbabilities tp = mp.extend(hypers.c0, hypers.c1, pi, pi0);
			pi = tp.pi;
			pi0 = tp.pi0;
			dd.addState();
			ed.addState();
			numStates++;
		}
		//		System.out.println();
	}

	private void cleanupStateSpace() {
		List<Integer> removedStates = mp.compress(ss);
		ed.removeStates(removedStates);
		dd.removeStates(removedStates);

		numStates -= removedStates.size();
	}

	/**
	 * 
	 * 
	 * @param minAux
	 * @return
	 */
	private ArrayList<LinkedHashMap<State,ArrayList<State>>> reachableStates(double minAux) {
		Range[] ranges = dd.durationRanges(minAux);
		ArrayList<LinkedHashMap<State,ArrayList<State>>> reachable = new ArrayList<LinkedHashMap<State,ArrayList<State>>>(auxArray.length);

		LinkedHashMap<State,ArrayList<State>> prevStates = new LinkedHashMap<State,ArrayList<State>>();
		prevStates.put(START_STATE, null);

		for(int t = 0; t < auxArray.length; t++) {
			double aux = auxArray[t];
			LinkedHashMap<State,ArrayList<State>> currStates = new LinkedHashMap<State,ArrayList<State>>();
			for (State prevState : prevStates.keySet()){
				if (prevState.getDuration() > 0) {
					State nextState = new State(prevState.getDuration()-1, prevState.getState());
					if (!currStates.containsKey(nextState)) {
						currStates.put(nextState, new ArrayList<State>());
					}
					currStates.get(nextState).add(prevState);
				} else {
					for (int s = 0; s < numStates; s++) {
						for (int d = ranges[s].min; d <= ranges[s].max; d++) {
							if (aux < fullTransitionProb(prevState.getState(), prevState.getDuration(), s, d)) {
								State nextState = new State(d, s);
								if (!currStates.containsKey(nextState)) {
									currStates.put(nextState, new ArrayList<State>());
								}
								currStates.get(nextState).add(prevState);
							}
						}
					}
				}
			}
			if (currStates.size() == 0) {
				return null;
			}
//			System.out.println(currStates.size() + " " + Runtime.getRuntime().freeMemory() / 1000000 + " " + Runtime.getRuntime().totalMemory() / 1000000 + " " + Runtime.getRuntime().maxMemory() / 1000000);
			reachable.add(currStates);
			prevStates = currStates;
		}

		return reachable;
	}

	/**
	 * 
	 * 
	 * @param reachableStates
	 * @return
	 */
	private ArrayList<HashMap<State, Double>> forwardPass(ArrayList<LinkedHashMap<State,ArrayList<State>>> reachableStates) {
		ArrayList<HashMap<State, Double>> alphas = new ArrayList<HashMap<State, Double>>(data.length+1);

		HashMap<State, Double> prevAlphas = new HashMap<State, Double>();
		prevAlphas.put(START_STATE, 1.0);

		for (int t = 0; t < data.length; t++) {
			LinkedHashMap<State,ArrayList<State>> reachable = reachableStates.get(t);
			HashMap<State, Double> currAlphas = new HashMap<State, Double>((int)(reachable.size()*1.5));
			double sum = 0;
			int i = 0;
			for (Map.Entry<State, ArrayList<State>> e : reachable.entrySet()) {
				State currState = e.getKey();
				double alpha = 0;
				for (State prevState : e.getValue()) {
					double p = auxDistr.probability(auxArray[t]/fullTransitionProb(prevState, currState));
					alpha += p*prevAlphas.get(prevState);
				}
				alpha *= ed.probability(currState.getState(), data[t]);
				sum += alpha;
				currAlphas.put(currState, alpha);
				i++;
			}
			if (sum == 0)
				return null;
			for (Map.Entry<State, Double> e : currAlphas.entrySet()) {
				e.setValue(e.getValue()/sum);
			}
			alphas.add(currAlphas);
			prevAlphas = currAlphas;
		}

		return alphas;
	}

	private void backwardSample(ArrayList<HashMap<State, Double>> alphas) {
		HashMap<State, Double> currAlphas;
		double rnd, sum, denom;
		for (int t = data.length - 1; t >= 0; t--) {
			currAlphas = alphas.get(t);
			if (t < data.length - 1) {
				denom = 0;
				for(Map.Entry<State,Double> e : currAlphas.entrySet()) {
					double p = fullTransitionProb(e.getKey(), ss[t+1]);
					if (auxArray[t+1] < p) {
						e.setValue(auxDistr.probability(auxArray[t+1]/p)*e.getValue());
						denom += e.getValue();
					} else {
						e.setValue(0.0);
					}
				}
			} else {
				denom = 1;
			}
			rnd = Util.RNG.nextDouble();
			sum = 0;
			for(Map.Entry<State,Double> e : currAlphas.entrySet()) {
				if (e.getValue() > 0) {
					sum += e.getValue()/denom;
					if (rnd < sum) {
						ss[t] = e.getKey();
						break;
					}
				}
			}
		}
	}

	private double sampleAuxiliaryVars() {
		double min = 1;
		State prev = START_STATE;
		for (int t = 0; t < auxArray.length; t++) {
			auxArray[t] = auxDistr.sample()*fullTransitionProb(prev, ss[t]);
			assert auxArray[t] != 0 : auxArray[t];
			assert !Double.isNaN(auxArray[t]) : auxArray[t];
			prev = ss[t];
			min = Math.min(min, auxArray[t]);
		}
		return min;
	}

	/**
	 * Sample gammas, pi, durations, emissions, and hyperparameters
	 * 
	 * @param obsCounts
	 */
	private void sampleParameters(int[][] obsCounts) {
		int[][] rCounts = sampleRCounts(obsCounts);
		mp.sampleParameters(hypers.c0, rCounts);
		TransitionProbabilities tp = mp.sampleTransitionMatrix(hypers.c1, obsCounts, ss[0].getState());  
		pi = tp.pi;
		pi0 = tp.pi0;

		dd.update(numStates, ss); 
		ed.update(numStates, ss, data);

		sampleHyperparameters(obsCounts); 
	}

	private static final int CONCENTRATION_SAMPLE_STD = 1;
	private static final Normal MH_RNG = new Normal(0, CONCENTRATION_SAMPLE_STD, Util.RNG);

	/**
	 * Sampling hyperparameters using Metropolis Hastings
	 * 
	 * @param obsCounts
	 */
	private void sampleHyperparameters(int[][] obsCounts) {
		double jll = jll(obsCounts);
		double c0 = hypers.c0;
		//		int accept = 0;
		for (int i = 0; i < MH_HYPER_ITERS; i++) {
			hypers.c0 = c0 + MH_RNG.nextDouble(); // sample
			if (hypers.c0 > 0) {
				double newJll = jll(obsCounts);
				if (newJll - jll >= 0 || Math.log(Util.RNG.nextDouble()) < newJll - jll) {
					//					accept++;
					jll = newJll;
					c0 = hypers.c0;
				}
			}
		}
		hypers.c0 = c0;
		//		System.out.printf("c0 accept rate = %f ", ((double)accept)/MH_HYPER_ITERS);
		//		accept = 0;

		double c1 = hypers.c1;
		for (int i = 0; i < MH_HYPER_ITERS; i++) {
			hypers.c1 = c1 + MH_RNG.nextDouble(); // sample
			if (hypers.c1 > 0) {
				double newJll = jll(obsCounts);
				if (Math.log(Util.RNG.nextDouble()) < newJll - jll) {
					//					accept++;
					jll = newJll;
					c1 = hypers.c1;
				}
			}
		}
		hypers.c1 = c1;
		//		System.out.printf("c1 accept rate = %f\n", ((double)accept)/MH_HYPER_ITERS);
	}
	

	private static final int SM_RESTRICTED_ITERS = 5;
	private int accepts = 0;
	private int total = 0;

	private void restrictedForwardBackward() {
		if (numStates <= 3)
			return;
		
		total++;
		Sample<P,E,D> oldConfig = new Sample<P,E,D>(pi0, pi, mp, ss, jll(calculateTransitionCounts()), hypers, null, dd.getParameters(), null, ed.getParameters(), null);

		int i = Math.abs(Util.RNG.nextInt()) % data.length;
		int j = i;
		while ((j = Math.abs(Util.RNG.nextInt()) % data.length) == i || ss[i].getState() == ss[j].getState()) 
			;
		
		int state1 = ss[i].getState();
		int state2 = ss[j].getState();
		Range[] r = dd.durationRanges(1e-4);
		int[] maxDurations = new int[ss.length];
		// calculate maxDurations
		for (int k = ss.length - 1; k >= 0; k--) {
			if (ss[k].getState() == state1 || ss[k].getState() == state2) {
				maxDurations[k] = k == ss.length - 1 ?  Math.max(r[state1].max, r[state2].max) : maxDurations[k+1]+1;
				//maxDuration = Math.max(maxDurations[k], maxDuration);
			} else {
				maxDurations[k] = -1;
			}
		}
		
		double[][][] alphas = restrictedForwardPass(maxDurations, state1, state2);
		double newLL;
		if (Double.isNaN((newLL = restrictedBackwardSample(alphas, maxDurations, state1, state2)))) {
			System.out.println("No restricted sampling path!");
			restore(oldConfig);
			return;
		}
		
		cleanupStateSpace();
		sampleTransitions();
		dd.update(numStates, ss);
		ed.update(numStates, ss, data);
		
		alphas = restrictedForwardPass(maxDurations, state1, state2);
		double oldLL = restrictedSampleLL(oldConfig.stateSequence, alphas, maxDurations, state1, state2);
		
		double newJLL = jll(calculateTransitionCounts());
		
		double logr = Math.log(Util.RNG.nextDouble());
		if (logr > (newJLL + oldLL - oldConfig.jll - newLL)) { // reject
			System.out.println("Rejected restricted forward-backward");
			// restore state
			restore(oldConfig);
		} else {
			System.out.println("Accepted restricted forward-backward");
			accepts++;
		}
	}
	
	/**
	 * 
	 */
	private void splitMerge() {
		total++;
		Sample<P,E,D> oldConfig = new Sample<P,E,D>(pi0, pi, mp, ss, jll(calculateTransitionCounts()), hypers, null, dd.getParameters(), null, ed.getParameters(), null);

		int i = Math.abs(Util.RNG.nextInt()) % data.length;
		int j = i;
		while ((j = Math.abs(Util.RNG.nextInt()) % data.length) == i) 
			;

		int state1, state2;
		int[] maxDurations = new int[ss.length];
		//int maxDuration = 0;
		boolean split = (ss[i].getState() == ss[j].getState()) || numStates == 2;
		if (split) {
			state1 = ss[i].getState();
			state2 = numStates;
			mp.extend(hypers.c0, hypers.c1, pi, pi0);
			dd.addState();
			ed.addState();
			numStates++;
			Range[] r = dd.durationRanges(1e-4);
			// move some states to the new state
			for (int k = ss.length - 1; k >= 0; k--) {
				if (ss[k].getState() == state1) {
					maxDurations[k] = Math.max(ss[k].getDuration(), k == ss.length - 1 ? Math.max(r[state1].max, r[state2].max) : maxDurations[k+1]+1);
					//maxDuration = Math.max(maxDurations[k], maxDuration);
					if (k != j && (k == i || Util.RNG.nextDouble() > .5)) {
						ss[k].setState(state2);
						if (k < ss.length - 1 && ss[k+1].getState() == state2) {
							ss[k].setDuration(ss[k+1].getDuration()+1);
						}
					}
				} else {
					maxDurations[k] = -1;
				}
			}
			sampleTransitions();
		} else { // merge: ss[i].getState() != ss[j].getState()
			state1 = ss[i].getState();
			state2 = ss[j].getState();
			Range[] r = dd.durationRanges(1e-4);
			// calculate maxDurations
			for (int k = ss.length - 1; k >= 0; k--) {
				if (ss[k].getState() == state1 || ss[k].getState() == state2) {
					maxDurations[k] = k == ss.length - 1 ?  Math.max(r[state1].max, r[state2].max) : maxDurations[k+1]+1;
					//maxDuration = Math.max(maxDurations[k], maxDuration);
				} else {
					maxDurations[k] = -1;
				}
			}
		}
		
		// start new code here //
		
		double[][][] alphas = restrictedForwardPass(maxDurations, state1, state2);
		double oldLL = restrictedSampleLL(oldConfig.stateSequence, alphas, maxDurations, state1, state2);
		double newLL;
		if (split) {
			if (Double.isNaN((newLL = restrictedBackwardSample(alphas, maxDurations, state1, state2)))) {
				System.out.println("No restricted sampling path!");
				restore(oldConfig);
				return;
			}
		} else {
			int keepState = Math.min(state1, state2);
			for (int t = ss.length - 1; t >= 0 ; t--) {
				if (alphas[t] != null) {
					ss[t].setState(keepState);
					if (t == ss.length - 1 || alphas[t+1] == null) {
						ss[t].setDuration(0);
					} else {
						ss[t].setDuration(ss[t+1].getDuration()+1);
					}
				}
			}
			newLL = restrictedSampleLL(ss, alphas, maxDurations, state1, state2);
		}
		
		cleanupStateSpace();
		sampleTransitions();
		dd.update(numStates, ss);
		ed.update(numStates, ss, data);
		
		
		
		// end new code here //
//		
//		for (int k = 0; k < SM_RESTRICTED_ITERS; k++) {
//			double[][][] alphas = restrictedForwardPass(maxDurations, state1, state2);
//			if (Double.isNaN(restrictedBackwardSample(alphas, maxDurations, state1, state2))) {
//				System.out.println("No restricted sampling path!");
//				restore(oldConfig);
//				return;
//			}
//			sampleTransitions();
//			dd.update(numStates, ss);
//			ed.update(numStates, ss, data);
//		}
//		
//		//Sample<P,E> launchConfig = new Sample<P,E>(pi0, pi, mp, ss, 0, hypers.c0, hypers.c1, dd.getParameters(), ed.getParameters());
//		
//		double[][][] alphas = restrictedForwardPass(maxDurations, state1, state2);
//		double oldLL = restrictedSampleLL(oldConfig.stateSequence, alphas, maxDurations, state1, state2);
//		double newLL;
//		if (split) {
//			if (Double.isNaN((newLL = restrictedBackwardSample(alphas, maxDurations, state1, state2)))) {
//				System.out.println("No restricted sampling path!");
//				restore(oldConfig);
//				return;
//			}
//		} else {
//			int keepState = Math.min(state1, state2);
//			for (int t = ss.length - 1; t >= 0 ; t--) {
//				if (alphas[t] != null) {
//					ss[t].setState(keepState);
//					if (t == ss.length - 1 || alphas[t+1] == null) {
//						ss[t].setDuration(0);
//					} else {
//						ss[t].setDuration(ss[t+1].getDuration()+1);
//					}
//				}
//			}
//			newLL = restrictedSampleLL(ss, alphas, maxDurations, state1, state2);
//		}
//		
//		cleanupStateSpace();
//		sampleTransitions();
//		dd.update(numStates, ss);
//		ed.update(numStates, ss, data);
		
		double newJLL = jll(calculateTransitionCounts());
		
		double logr = Math.log(Util.RNG.nextDouble());
		if (logr > (newJLL + oldLL - oldConfig.jll - newLL)) { // reject
			System.out.println("Rejected " + (split ? "split" : "merge"));
			// restore state
			restore(oldConfig);
		} else {
			System.out.println("Accepted " + (split ? "split" : "merge"));
			accepts++;
		}
	}
	
	private void restore(Sample<P,E,D> config) {
		pi0 = config.pi0;
		pi = config.pi;
		mp = config.mp;
		ss = config.stateSequence;
		dd.setParameters(config.durationParams);
		ed.setParameters(config.emitParams);
		numStates = pi.length;
	}

	/**
	 * 
	 * @param maxDuration
	 * @param maxDurations
	 * @param states
	 */
	private double[][][] restrictedForwardPass(int[] maxDurations, int... states) {
		double[][][] alphas = new double[ss.length][][];

		for (int t = 0; t < ss.length; t++) {
			if (maxDurations[t] >= 0) {
				alphas[t] = new double[states.length][];
				for (int s = 0; s < states.length; s++) {
					alphas[t][s] = new double[maxDurations[t]+1];
				}
			} else {
				alphas[t] = null;
			}
		}

		double sum = 0;
		// initialize if necessary
		if (alphas[0] != null) {
			for (int s = 0; s < states.length; s++) {
				for (int d = 0; d <= maxDurations[0]; d++) {
					alphas[0][s][d] = pi0[s]*dd.probability(states[s], d)*ed.probability(states[s], data[0]);
					sum += alphas[0][s][d];
				}
			}
			for (int s = 0; s < states.length; s++) {
				for (int d = 0; d <= maxDurations[0]; d++) {
					alphas[0][s][d] /= sum;
				}
			}
		}

		for (int t = 1; t < ss.length; t++) {
			if (alphas[t] != null) {
				sum = 0;
				if (alphas[t-1] != null) {
					for (int s1 = 0; s1 < states.length; s1++) {
						for (int d1 = 0; d1 <= maxDurations[t]; d1++) {
							for (int s2 = 0; s2 < states.length; s2++) {
								for (int d2 = 0; d2 <= maxDurations[t]; d2++) {
									alphas[t][s1][d1] += fullTransitionProb(states[s2], d2, states[s1], d1)*alphas[t-1][s2][d2];
								}
							}
							alphas[t][s1][d1] *= ed.probability(states[s1], data[t]);
							sum += alphas[t][s1][d1];
						}
					}
				} else { // maxDurations[t-1] == -1
					for (int s = 0; s < states.length; s++) {
						for (int d = 0; d <= maxDurations[t]; d++) {
							alphas[t][s][d] = fullTransitionProb(ss[t-1].getState(), ss[t-1].getDuration(), states[s], d)*ed.probability(states[s], data[t]);
							sum += alphas[t][s][d];
						}
					}
				}
				for (int s = 0; s < states.length; s++) {
					for (int d = 0; d <= maxDurations[t]; d++) {
						alphas[t][s][d] /= sum;
					}
				}
			}
		}

		return alphas;
	}
	
	private double restrictedBackwardSample(double[][][] alphas, int[] maxDurations, int... states) {
		double ll = 0;
		double sum = 0;
		double r;
		int last = ss.length - 1;
		
		if (alphas[last] != null) {
			r = Util.RNG.nextDouble();
			OUTSIDE:
				for (int s = 0; s < states.length; s++) {
					for (int d = 0; d <= maxDurations[last]; d++) {
						sum += alphas[last][s][d];
						if (r < sum) {
							ss[last] = new State(d,states[s]);
							break OUTSIDE;
						}
					}
				}
		}
		
		for (int t = last - 1 ; t >= 0; t--) {
			if (alphas[t] != null) {
				double denom = 0;
				for (int s = 0; s < states.length; s++) {
					for (int d = 0; d <= maxDurations[t]; d++) {
						alphas[t][s][d] *= fullTransitionProb(states[s],d,ss[t+1].getState(), ss[t+1].getDuration());
						denom += alphas[t][s][d];
					}
				}
				if (denom == 0) {
					return Double.NaN;
				}
				r = Util.RNG.nextDouble();
				sum = 0;
				OUTSIDE:
					for (int s = 0; s < states.length; s++) {
						for (int d = 0; d <= maxDurations[t]; d++) {
							sum += alphas[t][s][d]/denom;
							if (r < sum) {
								ss[t] = new State(d, states[s]);
								ll += Math.log(alphas[t][s][d]);
								break OUTSIDE;
							}
						}
					}
			}
		}
		return ll;
	}
	
	private double restrictedSampleLL(State[] ss, double[][][] alphas, int[] maxDurations, int... states) {
		HashMap<Integer,Integer> reverseStateMap = new HashMap<Integer, Integer>();
		for (int s = 0; s < states.length; s++) {
			reverseStateMap.put(states[s], s);
		}
		double ll = 0;
		for (int t = ss.length - 1 ; t >= 0; t--) {
			if (alphas[t] != null) {
				double l = Math.log(alphas[t][reverseStateMap.get(ss[t].getState())][ss[t].getDuration()]);
				ll += l;
			}
		}
		return ll;
	}

	private void sampleTransitions() {
		int[][] obsCounts = calculateTransitionCounts();
		int[][] rCounts = sampleRCounts(obsCounts);
		mp.sampleParameters(hypers.c0, rCounts);
		TransitionProbabilities tp =  mp.sampleTransitionMatrix(hypers.c1, obsCounts, ss[0].getState());
		pi = tp.pi;
		pi0 = tp.pi0;
	}

	private int[][] calculateTransitionCounts() {
		return calculateTransitionCounts(ss);
	}
			
	/**
	 * 
	 * @return
	 */
	private int[][] calculateTransitionCounts(State[] ss) {
		int[][] obsCounts = new int[numStates][numStates+1];

		for (int t = 1; t < ss.length; t++) {
			if (ss[t-1].getDuration() == 0) {
				assert ss[t-1].getState() != ss[t].getState();
				obsCounts[ss[t-1].getState()][ss[t].getState()]++;
			}
		}
		return obsCounts;
	}

	/**
	 * 
	 * @param obsCounts
	 * @return
	 */
	private int[][] sampleRCounts(int[][] obsCounts) {
		int[][] rCounts = new int[obsCounts.length][obsCounts.length + 1];
		for (int m = 0; m < rCounts.length; m++) {
			for (int k = 0; k < rCounts[m].length; k++) {
				double c1b = hypers.c1 * mp.getBeta(m, k);
				for (int i = 0; i < obsCounts[m][k]; i++) {
					if (Util.RNG.nextDouble() < c1b / (c1b + i)) {
						rCounts[m][k]++;
					}
				}
			}
		}

		return rCounts;
	}

	private double fullTransitionProb(State oldS, State newS) {
		return fullTransitionProb(oldS.getState(), oldS.getDuration(), newS.getState(), newS.getDuration());
	}
	/**
	 * 
	 * 
	 * @param oldS
	 * @param oldD
	 * @param newS
	 * @param newD
	 * @return
	 */
	private double fullTransitionProb(int oldS, int oldD, int newS, int newD) {
		if (oldD > 0) {
			return (newD == oldD - 1 && oldS == newS) ? 1 : 0; 
		} else if (oldS == START_STATE.getState()) {
			return pi0[newS]*dd.probability(newS, newD);
		}
		return pi[oldS][newS]*dd.probability(newS, newD);
	}
	
	/**
	 * Calculate the current joint log likelihood of the model
	 * 
	 * @return the joint log likelihood
	 */
	public double jll() {
		return jll(calculateTransitionCounts());
	}

	private double jll(int[][] obsCounts) { 
		return jll(obsCounts, false);
	}

	private double jll(int[][] obsCounts, boolean print) {
		// emissions and duration LLs
		double jll = ed.logLikelihood(ss, data);
		if (print) System.out.println(jll);
		jll += dd.logLikelihood(ss);
		if (print) System.out.println(jll);

		assert !Double.isInfinite(jll) && !Double.isNaN(jll) : jll + "\n" + ed + "\n" + dd;

		// transition LL
		for (int m = 0; m < obsCounts.length; m++) {
			double[] c1b = mp.getBetas(m);
			Util.multiply(hypers.c1, c1b);
			double ll = DirichletDistribution.logPartition(Util.addArrays(c1b, obsCounts[m])) - DirichletDistribution.logPartition(c1b);
			//jll += DirichletDistribution.logPartition(Util.addArrays(c1b, obsCounts[m]));
			//jll -= DirichletDistribution.logPartition(c1b);
			jll += ll;
			assert !Double.isInfinite(jll) && !Double.isNaN(jll) : jll + " " + Arrays.toString(c1b);
		}
		if (print) System.out.println(jll);

		jll += mp.gammasLogLikelihood(hypers.c0);
		assert !Double.isInfinite(jll) && !Double.isNaN(jll) : jll + "\n" + mp;
		if (print) System.out.println(jll);
		jll += (numStates+1)*GammaDistribution.logLikelihood(hypers.c0_a, hypers.c0_b, hypers.c0);
		assert !Double.isInfinite(jll) && !Double.isNaN(jll) : jll;
		jll += numStates*GammaDistribution.logLikelihood(hypers.c1_a, hypers.c1_b, hypers.c1);
		if (print) System.out.println(jll);

		assert !Double.isInfinite(jll) && !Double.isNaN(jll) : jll;
		return jll;
	}
	
	/**
	 * Sample a state sequence using the current configuration
	 * of the ED-iHMM
	 * 
	 * @param len
	 * @param states
	 * @return
	 */
	private State[] sampleStateSequence(int len, int states) {
		State[] ss = new State[len];
		State prevState = null;
		for (int i = 0; i < len; i++) {
			ss[i] = sampleNextState(prevState);
			prevState = ss[i];
		}
		System.out.println(Arrays.toString(ss));
		return ss;
	}

	/**
	 * Sample the next state using the current configuration
	 * of the ED-iHMM
	 * 
	 * @param curr
	 * @return
	 */
	private State sampleNextState(State curr) {
		assert curr == null || curr.getDuration() >= 0;

		State next = curr != null ? (State)curr.clone() : new State(0,0);

		if (next.getDuration() == 0) {
			double[] trans = curr == null ? pi0 : pi[next.getState()];
			next.setState(CategoricalDistribution.sample(trans));
			if (next.getState() == pi.length) {
				System.out.println("extending state space during sampling");
				dd.addState();
				ed.addState();
				TransitionProbabilities tp = mp.extend(hypers.c0, hypers.c1, pi, pi0);
				pi = tp.pi;
				pi0 = tp.pi0;
				numStates++;
			}
			next.setDuration(dd.sample(next.getState()));
		} else {
			next.setDuration(next.getDuration()-1);
		}
		return next;
	}

	public void addIterationListener(IterationListener il) {
		iterListeners.add(il);
	}

	public void removeIterationListener(IterationListener il) {
		iterListeners.remove(il);
	}

	private void passSampleToListeners(Sample<P,E,D> s, boolean keep) {
		for(IterationListener il : iterListeners) {
			il.newSample(s, keep);
		}
	}

}
