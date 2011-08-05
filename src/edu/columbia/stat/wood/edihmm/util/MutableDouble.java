package edu.columbia.stat.wood.edihmm.util;

/**
 * @author nicholasbartlett
 */
public class MutableDouble {

    private double value;

    public MutableDouble(double value) {
        this.value = value;
    }

    public double doubleValue() {
        return value;
    }

    public void set(double value) {
        this.value = value;
    }

    public void plusEquals(double adjustment) {
        value += adjustment;
    }

    public void timesEquals(double factor) {
        value *= factor;
    }
    
    public static MutableDouble[] toMutableArray(double[] doubles) {
    	MutableDouble[] mdoubles = new MutableDouble[doubles.length];
    	for (int i = 0; i < doubles.length; i++)
    		mdoubles[i] = new MutableDouble(doubles[i]);
    	return mdoubles;
    }

    public void addLogs(double log_something) {
        if (Double.isInfinite(value) && Double.isInfinite(log_something)) {
            if (value < 0d && log_something < 0d) {
                value = Double.NEGATIVE_INFINITY;
            } else {
                throw new RuntimeException("basically shouldn't happen");
            }
        } else if (value > log_something) {
            value = Math.log(1.0 + Math.exp(log_something - value)) + value;
        } else {
            value = Math.log(1.0 + Math.exp(value - log_something)) + log_something;
        }
        if (Double.isNaN(value)) {
            System.out.println();
        }
    }
}
