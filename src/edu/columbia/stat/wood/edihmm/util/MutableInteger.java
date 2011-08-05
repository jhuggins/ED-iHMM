/*
 * Created on Jun 30, 2011
 */
package edu.columbia.stat.wood.edihmm.util;

/**
 * @author Jonathan Huggins
 *
 */
public class MutableInteger {

	private int value;

    public MutableInteger(int value) {
        this.value = value;
    }

    public int intValue() {
        return value;
    }

    public void set(int value) {
        this.value = value;
    }

    public void plusEquals(int adjustment) {
        value += adjustment;
    }

    public void timesEquals(int factor) {
        value *= factor;
    }
	
}
