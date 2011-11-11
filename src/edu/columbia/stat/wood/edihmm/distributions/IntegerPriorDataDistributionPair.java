package edu.columbia.stat.wood.edihmm.distributions;


import edu.columbia.stat.wood.edihmm.util.Range;

/**
 * 
 * @author Jonathan Huggins
 *
 * @param <P> 
 */
public abstract class IntegerPriorDataDistributionPair<P> extends PriorDataDistributionPair<P, Integer> {
	
	
	private static final long serialVersionUID = -1798287353718472274L;
	private static final int RANGE_SEARCH_INCR = 1;
	/**
	 * 
	 * Constructs a <tt>IntegerPriorDataDistributionPair</tt> object.
	 *
	 * @param priorDistr
	 * @param dataDistr
	 */
	protected IntegerPriorDataDistributionPair(Distribution<P> priorDistr) {
		super(priorDistr);
	}


	/**
	 * 
	 * @param param
	 * @param minLL
	 * @return
	 */
	public Range range(P param, double minLL) {
		if (dataLogLikelihood(param, mode(param)) < minLL) {
			return new Range(0,-1);
		}
		int min = -1;
		int max = -1;
		double dll;
		for (int i = 0; ; i += RANGE_SEARCH_INCR) {
//			System.out.println(i);
			dll = dataLogLikelihood(param, i);
			if (min == -1 && dll > minLL) {
				min = i;
			} else if(min > -1 && dll < minLL) {
				max = i - 1;
				break;
			}
		}
	
		return new Range(min, max);
	}
	
	protected abstract int mode(P param);


}
