package edu.columbia.stat.wood.edihmm.distributions;

import java.io.Serializable;

public class OffsetGeometricParams implements Serializable {

	private static final long serialVersionUID = -8994298988880354917L;
	
	public int offset;
	public double p;
	
	public OffsetGeometricParams(int offset, double p) {
		this.offset = offset;
		this.p = p;
	}
	
	public String toString() {
		return "[" + offset + " " + p + "]";
	}
	
}
