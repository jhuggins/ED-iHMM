package edu.columbia.stat.wood.edihmm;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Set;

import cern.jet.stat.Gamma;

import edu.columbia.stat.wood.edihmm.distributions.BetaDistribution;
import edu.columbia.stat.wood.edihmm.distributions.DirichletDistribution;
import edu.columbia.stat.wood.edihmm.distributions.GammaDistribution;
import edu.columbia.stat.wood.edihmm.util.Util;

/**
 * Holds all the information about the underlying Gamma Process,
 * such as weighting information and where states live in the
 * augmented space. Handles sampling of the GammaP and the transition
 * matrix.
 * 
 * @author Jonathan Huggins
 *
 */
public class MixingProportions implements Serializable, Cloneable {

	private static final long serialVersionUID = -7512263931718029423L;

	public static enum Mode { GEOM, POISSON };
	private static final int GAMMA_SAMPLE_ITERS = 5;
	/**
	 * The gamma values are the basis for the mixing proportions. Note that
	 * the last one corresponds to the "extra" gamma_+ value
	 */
	private List<Double> gamma;
	/**
	 * Sum of all the gamma components
	 */
	private double gammaSum;
	/**
	 * The augmented variables
	 */
	private List<Integer> indices;
	/**
	 * The augmented variables, as a set
	 */
	private Set<Integer> indSet;
	/**
	 * Probability of drawing an unobserved index variable
	 */
	private double unobsIndProb;
	/**
	 * Parameter for the geometric/poisson distribution from which
	 * the index variables are drawn
	 */
	public double q;
	private double a;
	private double b;
	private Mode mode;
	private boolean sampleQ;
	/**
	 * Auxiliary variables for sampling; length is one less than the length of gamma
	 */
	private List<Double> aux;
	/**
	 * Sum of all the auxiliary variables
	 */
	private double auxSum;
	
	/**
	 * 
	 * Constructs a <tt>MixingProportions</tt> object. <tt>q</tt> is not sampled.
	 *
	 * @param c0
	 * @param rCounts
	 * @param q
	 * @param mode
	 */
	public MixingProportions(double c0, int[][] rCounts, double q, Mode mode) {
		this(c0, rCounts, q, 1, 1, mode);
	}
	
	/**
	 * 
	 * Constructs a <tt>MixingProportions</tt> object, using <tt>c0</tt> and
	 * <tt>rCounts</tt> to initially sample gamma and the auxiliary variables
	 * 
	 * <tt>a</tt> and <tt>b</tt> are the parameters for the scale distribution,
	 * which is the prior for the parameter <tt>q</tt> for the geometric 
	 * distribution from which elements of the index list are drawn.
	 *
	 * @param c0
	 * @param rCounts
	 * @param a 
	 * @param b 
	 * @throws MathException
	 */
	public MixingProportions(double c0, int[][] rCounts, double a, double b, Mode mode) {
		this(c0, rCounts, 0, a, b, mode);
	}
	
	private MixingProportions() {
		
	}
	
	/**
	 * 
	 * Constructs a <tt>MixingProportions</tt> object, using <tt>c0</tt> and
	 * <tt>rCounts</tt> to initially sample gamma and the auxiliary variables
	 * 
	 * <tt>a</tt> and <tt>b</tt> are the parameters for the scale distribution,
	 * which is the prior for the parameter <tt>q</tt> for the geometric 
	 * distribution from which elements of the index list are drawn.
	 *
	 * @param c0
	 * @param rCounts
	 * @param a 
	 * @param b 
	 * @throws MathException
	 */
	private MixingProportions(double c0, int[][] rCounts, double q, double a, double b, Mode mode) {
		assert rCounts.length == rCounts[0].length - 1;
		
		if (a <= 0 || b <= 0 || q >= 1) {
			throw new IllegalArgumentException("MixingProportions: a and b must be greater than 0 and q must be less than 1");
		}
		
		int cap = 2*rCounts[0].length;
		gamma = new ArrayList<Double>(cap);
		aux = new ArrayList<Double>(cap);
		indices = new ArrayList<Integer>(cap);
		indSet = new HashSet<Integer>(cap);
		this.a = a;
		this.b = b;
		this.mode = mode;
		
		System.out.println ("q = " + q);
		sampleQ = q <= 0;
		if (sampleQ) {
				this.q = BetaDistribution.sample(a, b);
		} else {
			this.q = q;
		}
		unobsIndProb = 1;
		for (int i = 0; i < rCounts.length; i++) {
			gamma.add(0.0);
			aux.add(0.0);
			extendIndexList();
		}
		gamma.add(0.0);
		gammaSum = 0;
		auxSum = 0;
		
		// initial sampling
		sampleParameters(c0, rCounts);
	}
	
	public int states() {
		return gamma.size()-1;
	}
	
	public double gammasLogLikelihood(double c0) {
//		double ll = 0;
//		System.out.println(ll);
//		for (int i = 0; i < gamma.size()-1; i++) {
//			ll += GammaDistribution.logLikelihood(c0*indexProb(indices.get(i)), 1, gamma.get(i));
//			System.out.println(ll);
//		}
//		ll += GammaDistribution.logLikelihood(c0*unobsIndProb, 1, gamma.get(indices.size()));
//		System.out.println(ll);
//		return ll;
		double[] params = new double[gamma.size()];
		for (int i = 0; i < indices.size(); i++) {
			params[i] = c0*indexProb(indices.get(i));
		}
		params[indices.size()] = c0*unobsIndProb;
		double ll = DirichletDistribution.logLikelihood(params, getNormalizedGammas());
//		System.out.println(ll);
		assert !Double.isInfinite(ll) && !Double.isNaN(ll) : ll + " " + Arrays.toString(params) + " " + Arrays.toString(getNormalizedGammas());
		return ll;
	}
	
	/**
	 * Get the k-th mixing proportion for the m-th dependent DP
	 * 
	 * 
	 * @param m current state
	 * @param k next state
	 * @return
	 */
	public double getBeta(int m, int k) {
		assert 0 <= m && m < gamma.size() && 0 <= k && k < gamma.size();
		return m == k ? 0 : gamma.get(k) / (gammaSum - gamma.get(m)); 
	}
	
	
	/**
	 * Get the m-th row of scale ÐÐ i.e., the mixing proportions for 
	 * the m-th dependent DP
	 * 
	 * @param m
	 * @return
	 */
	public double[] getBetas(int m) {
		double[] betas = new double[gamma.size()];
		for (int k = 0; k < betas.length; k++) { 
			betas[k] = getBeta(m, k);
		}
		return betas;
	}
	
	public double[] getNormalizedGammas() {
		double[] normg = new double[gamma.size()];
		for (int k = 0; k < normg.length; k++) { 
			normg[k] = gamma.get(k) / gammaSum;
		}
		return normg;
	}
	
	/**
	 * Counts should have zeros in the last column
	 * 
	 * @param c1
	 * @param obsCounts observed transition counts
	 * @return 
	 */
	public TransitionProbabilities sampleTransitionMatrix(double c1, int[][] obsCounts) {
		return sampleTransitionMatrix(c1, obsCounts, null);
	}
	/**
	 * Counts should have zeros in the last column
	 * 
	 * @param c1
	 * @param obsCounts observed transition counts
	 * @return 
	 */
	public TransitionProbabilities sampleTransitionMatrix(double c1, int[][] obsCounts, List<State[]> ss) {
		int[] startStates = new int[ss == null ? 0 : ss.size()];
		if (ss != null) {
			int i = 0;
			for (State[] s : ss) {
				startStates[i++] = s[0].getState();
			}
		}
		double[][] pi = new double[gamma.size() - 1][gamma.size()];
		for (int m = 0; m < pi.length; m++) {
			assert obsCounts[m][m] == 0 : "Diagonal of counts matrix should be all zeros";
			double[] betas = getBetas(m);
			Util.multiply(c1, betas);
			pi[m] = DirichletDistribution.sample(Util.addArrays(betas, obsCounts[m]));
		}
		return new TransitionProbabilities(pi, samplePi0(c1, startStates));
	}
	
	/**
	 * 
	 * @param c1
	 * @param startStates
	 * @return
	 */
	private double[] samplePi0(double c1, int... startStates) {
		double[] params = getNormalizedGammas();
		Util.multiply(c1, params);
		for (int s : startStates) {
			params[s]++;
		}
		return DirichletDistribution.sample(params);
	}
	
	/**
	 * Sample indices and gammas
	 * 
	 * @param c0
	 * @param rCounts
	 */
	public void sampleParameters(double c0, int[][] rCounts) {
		assert aux.size() == gamma.size() - 1 : String.format("%d, %d", aux.size(), gamma.size());
		assert indices.size() == gamma.size() - 1 : String.format("%d, %d", indices.size(), gamma.size());
		assert indices.size() == indSet.size() : String.format("%d, %d", indices.size(), indSet.size());
		assert rCounts.length == rCounts[0].length - 1;
		assert rCounts.length == gamma.size() - 1;
		assert q > 0;
		
		if (sampleQ) {
			sampleQ(rCounts);
		}
		for (int i = 0; i < GAMMA_SAMPLE_ITERS; i++) {
			sampleGamma(c0, rCounts);
		}
	}
	
	/**
	 * Sample gamma 
	 * 
	 * @param rCounts restaurant counts
	 */
	private void sampleGamma(double c0, int[][] rCounts) {
		double s;
		// sample auxiliary variables
		auxSum = 0;
		for (int i = 0; i < aux.size(); i++) {
			assert rCounts[i][i] == 0 : "Diagonal of counts matrix should be all zeros: " + i;
			s = gammaSum > 0 ? GammaDistribution.sample(Util.sum(rCounts[i]) , 1.0 / (gammaSum - gamma.get(i))) : 0;
			aux.set(i, s);
			auxSum += s;
		}
		
		// sample gamma
		gammaSum = 0;
		for (int m = 0; m < gamma.size()-1; m++) {
			int cMarginal = Util.sumColumn(rCounts, m); 
			s = GammaDistribution.sample(cMarginal + indexProb(indices.get(m)), 1.0/(1 + auxSum - aux.get(m)));
			gamma.set(m, s);
			gammaSum += s;
		}
		s = GammaDistribution.sample(c0*unobsIndProb, 1.0 / (1 + auxSum));
		gamma.set(gamma.size()-1, s);
		gammaSum += s;
	}
	
	private void sampleQ(int[][] rCounts) {
		// sample q
		int obsCount = 0;
		int sum = 0;
		for (int i = 0; i < rCounts.length; i++) {
			for (int j = 0; j < rCounts[i].length; j++) {
				sum += rCounts[i][j];
				obsCount += rCounts[i][j]*indices.get(i);
			}
		}
		switch(mode) {
		case GEOM:
			q = BetaDistribution.sample(a + obsCount, b + sum);
			break;
		case POISSON:
			q = GammaDistribution.sample(a + sum, b/(obsCount*b + 1));
			break;
		}
		// update total unobserved index probability
		unobsIndProb = 1;
		for (int i : indices) {
			unobsIndProb -= indexProb(i);
		}
	}
	
	/**
	 * For now, a very dumb implementation that assumes 
	 * the indices are geometrically distributed
	 * 
	 * TODO: improve
	 * 
	 * @return
	 */
	private int sampleNewIndexVar() {
		double rnd = Util.RNG.nextDouble()*unobsIndProb;
		double sum = 0;
		//System.out.print("new ind: ");
		for (int i = 1; ; i++) {
			//System.out.print(i + " ");
			if (!indSet.contains(i)) {
				sum += indexProb(i);
				if (rnd < sum) {
					//System.out.println("*");
					return i;
				}
			}
		}
		
	}
	
	/**
	 * Sample a new index, add it to the list and set, and 
	 * update the total unobserved index probability
	 * 
	 * @return probability of the new index
	 */
	private double extendIndexList() {
		int augSamp = sampleNewIndexVar();
		indices.add(augSamp);
		indSet.add(augSamp);
		double p = indexProb(augSamp);
		//System.out.printf("new prob = %f; unobs prob = %f\n", p, unobsIndProb);
		unobsIndProb -= p;
		return p;
	}
	
	/**
	 * 
	 * 
	 * @param ind
	 * @return
	 */
	private double indexProb(int ind) {
		switch(mode) {
		case GEOM:
			return (1-q)*Math.pow(q, ind-1);
		case POISSON:
			return Math.exp((ind-1)*Math.log(q)-q-Gamma.logGamma(ind));
		}
		throw new RuntimeException("MixingProportions: invalid mode");
	}
	
	/**
	 * Extend the length of gamma by one, and extend
	 * <tt>pi</tt> if it is non-null. 
	 * 
	 * @param c0
	 * @param pi
	 */
	public TransitionProbabilities extend(double c0, double c1, double[][] pi, double[] pi0) {
		assert q > 0;
		assert pi.length == pi[0].length - 1;
		assert pi.length == gamma.size() - 1;
		assert aux.size() == gamma.size() - 1 : String.format("%d, %d", aux.size(), gamma.size());
		assert indices.size() == gamma.size() - 1 : String.format("%d, %d", indices.size(), gamma.size());
		assert indices.size() == indSet.size() : String.format("%d, %d", indices.size(), indSet.size());
		
		int n = gamma.size()-1;  // index of row/column of new state
		double gplus = gamma.get(n);
		double brk = BetaDistribution.sample(1, c0*unobsIndProb);
		double g = gplus*(1-brk);
		gplus = gplus*brk;
		
		extendIndexList();

		gamma.add(gplus); // move gamma_+ over one index
		gamma.set(n, g); // insert gamma_n directly before gamma_+
		// N.B. gammaSum stays the same
		aux.add(0.0);

		assert aux.size() == gamma.size() - 1 : String.format("%d, %d", aux.size(), gamma.size());
		assert indices.size() == gamma.size() - 1 : String.format("%d, %d", indices.size(), gamma.size());
		assert indices.size() == indSet.size() : String.format("%d, %d", indices.size(), indSet.size());
		
		if (pi != null && pi0 != null) {
			// create an extended pi matrix
			double[][] newPi = new double[pi.length+1][pi[0].length+1];
			double[] newPi0 = new double[pi0.length+1];
			for (int i = 0; i < pi.length; i++) {
				System.arraycopy(pi[i], 0, newPi[i], 0, pi[i].length - 1);
			}
			System.arraycopy(pi0, 0, newPi0, 0, pi0.length);
			// add new row
			double[] betas = getBetas(n);
			Util.multiply(c1, betas);
			newPi[n] = DirichletDistribution.sample(betas);
			
			// add new column of probabilities & update unobserved transition probabilities
			for (int i = 0; i < n; i++) {
				//System.out.printf("scale(%d) = %s\n", i, Arrays.toString(getBetas(i)));
				double prop = BetaDistribution.sample(c1*getBeta(i,n), c1*getBeta(i,n+1));
				//System.out.printf("scale(%d,n)=%f, scale(%d,n+1)=%f, prop=%f\n", i, getBeta(i,n), i, getBeta(i,n+1), prop);
				newPi[i][n] = prop*pi[i][n];
				newPi[i][n+1] = (1-prop)*pi[i][n];
			}
			
			double prop = BetaDistribution.sample(c1*gamma.get(n)/gammaSum, c1*gamma.get(n+1)/gammaSum);
			newPi0[n] = prop*pi0[n];
			newPi0[n+1] = (1-prop)*pi0[n];
			
			pi = newPi;
			pi0 = newPi0;
		}
		return new TransitionProbabilities(pi, pi0);
	}
	
	/**
	 * Remove states if they are unused. Update <tt>stateSeq</tt> to
	 * reflect these changes. 
	 * 
	 * @param counts
	 * @return list of removed states
	 */
	public List<Integer> compress(List<State[]> seqs) {
		assert q > 0;
		assert aux.size() == gamma.size() - 1 : String.format("%d, %d", aux.size(), gamma.size());
		assert indices.size() == gamma.size() - 1 : String.format("%d, %d", indices.size(), gamma.size());
		assert indices.size() == indSet.size() : String.format("%d, %d", indices.size(), indSet.size());
		
		boolean[] didObserveState = new boolean[states()];
		for (State[] seq : seqs) {
			for (State s : seq) {
				didObserveState[s.getState()] = true;
			}
		}
		LinkedList<Integer> removedStates = new LinkedList<Integer>();
		int[] offsets = new int[states()];
		int offset = 0;
		for (int i = 0; i < didObserveState.length; i++) { 
			if (!didObserveState[i]) {
				int j = i + offset; // adjust for already removed states
				removedStates.add(i);
				gammaSum -= gamma.get(j);
				auxSum -= aux.get(j);
				gamma.remove(j);
				aux.remove(j);
				Integer ind = indices.remove(j);
				indSet.remove(ind);
				unobsIndProb += indexProb(ind);
				offset--;
			} else {
				offsets[i] = offset;
			}
		}
		// adjust state values in the state sequence
		if (offset < 0) {
			for (State[] seq : seqs) {
				for (State s : seq) {
					s.setState(s.getState() + offsets[s.getState()]);
				}
			}
		}
		return removedStates;
	}
	
	@Override
	public String toString() {
		return String.format("MixingProportions(states=%s, gamma=%s, indices=%s, q=%f, unobsIndProb=%f, (a,b) = (%f,%f))",
				states(), gamma, indices, q, unobsIndProb, a, b);
	}
	
	/**
	 * A deep copy of this object
	 */
	@Override
	public MixingProportions clone() {
		MixingProportions mp = new MixingProportions();
		 mp.gamma = new ArrayList<Double>(gamma);
		mp.gammaSum = gammaSum;
		mp.indices = new ArrayList<Integer>(indices);
		mp.indSet = new HashSet<Integer>(indSet);
		mp.unobsIndProb = unobsIndProb;
		mp.q = q;
		mp.a = a;
		mp.b = b;
		mp.mode = mode;
		mp.sampleQ = sampleQ;
		mp.aux = new ArrayList<Double>(aux);
		mp.auxSum = auxSum;
		return mp;
	}
}
