/*
 * Created on Jun 30, 2011
 */
package edu.columbia.stat.wood.edihmm.util;

/**
 * @author Jonathan Huggins
 *
 */
public class Range {

	public int min;
	public int max;
	
	/**
	 * Constructs a <tt>Range</tt> object.
	 *
	 * @param min
	 * @param max
	 */
	public Range(int min, int max) {
		this.min = min;
		this.max = max;
	}
	
	public void extendRange(Range other) {
		min = Math.min(min, other.min);
		max = Math.max(max, other.max);
	}
	
	@Override
	public String toString() {
		return "Range(min=" + min + ", max=" + max + ")";
	}
	
}
