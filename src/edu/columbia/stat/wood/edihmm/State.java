/*
 * Created on Jun 29, 2011
 */
package edu.columbia.stat.wood.edihmm;

import java.io.Serializable;

/**
 * The full state of the ED-iHMM, consisting of a remaining 
 * duration and the state's index. 
 * 
 * @author Jonathan Huggins
 *
 */ 
public class State implements Cloneable, Serializable {

	private static final long serialVersionUID = 4912096056885029192L;
	
	private int duration;
	private int state;
	
	/**
	 * Constructs a <tt>State</tt> object.
	 *
	 * @param duration
	 * @param state
	 */
	public State(int duration, int state) {
		this.duration = duration;
		this.state = state;
	}

	public State(State s) {
		this.duration = s.duration;
		this.state = s.state;
	}

	/**
	 * @return the duration
	 */
	public int getDuration() {
		return duration;
	}
	
	/**
	 * Set the duration.
	 * 
	 * @param duration
	 */
	public void setDuration(int duration) {
		this.duration = duration;
	}

	/**
	 * @return the state
	 */
	public int getState() {
		return state;
	}
	
	/**
	 * Set the state.
	 * 
	 * @param state
	 */
	public void setState(int state) {
		this.state = state;
	}
	
	@Override
	public State clone() {
		return new State(duration, state);
	}
	
	@Override
	public String toString() {
		return String.format("State(state=%d,duration=%d)", state, duration);
	}
	
	@Override
	public int hashCode() {
		return state ^ (duration << 4);
	}
	
	@Override 
	public boolean equals(Object o) {
		if (o instanceof State) {
			State s = (State)o;
			return s.getDuration() == duration && s.getState() == state;
		}
		return false;
	}
	
}
