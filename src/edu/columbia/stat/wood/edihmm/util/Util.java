/*
 * Created on Jun 29, 2011
 */
package edu.columbia.stat.wood.edihmm.util;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import cern.jet.random.engine.MersenneTwister;
import cern.jet.random.engine.RandomEngine;
import edu.columbia.stat.wood.edihmm.DurationDistribution;
import edu.columbia.stat.wood.edihmm.EDiHMM;
import edu.columbia.stat.wood.edihmm.EmissionDistribution;
import edu.columbia.stat.wood.edihmm.Sample;
import edu.columbia.stat.wood.edihmm.State;
import edu.columbia.stat.wood.edihmm.distributions.GaussianParams;
import edu.columbia.stat.wood.edihmm.distributions.MultivariateGaussianParams;
import gov.sandia.cognition.math.matrix.Matrix;
import gov.sandia.cognition.math.matrix.MatrixFactory;
import gov.sandia.cognition.math.matrix.Vector;
import gov.sandia.cognition.math.matrix.VectorFactory;
import gov.sandia.cognition.math.matrix.mtj.DenseMatrixFactoryMTJ;
import gov.sandia.cognition.math.matrix.mtj.DenseVectorFactoryMTJ;

/**
 * @author Jonathan Huggins
 *
 */
public class Util {

	public static final Random RAND = new Random(10);
	//public static final RandomEngine RNG = new MersenneTwister(new Date());
	public static RandomEngine RNG = new MersenneTwister(11);
	//public static final double LN_2 = Math.log(2);
	public static final double EPS = 1e-100;
	
	public static final VectorFactory<? extends Vector> VF = DenseVectorFactoryMTJ.getDenseDefault();
	public static final MatrixFactory<? extends Matrix> MF = DenseMatrixFactoryMTJ.getDenseDefault();
	
	public static double[] addArrays(double[] da, int[] ia) {
		assert da.length == ia.length;
		double[] sum = new double[da.length];
		for (int i = 0; i < sum.length; i++) {
			sum[i] = da[i] + ia[i];
		}
		return sum;
	}
	
    public static int sum(int[] arr) {
        int s = 0;
        if (arr != null) {
            for (int i = 0; i < arr.length; i++) {
                s += arr[i];
            }
        }
        return s;
    }

    public static double sum(double[] arr) {
        double s = 0;
        if (arr != null) {
            for (int i = 0; i < arr.length; i++) {
                s += arr[i];
            }
        }
        return s;
    }
    
    public static int sumColumn(int[][] arr, int col) {
    	int sum = 0;
    	for (int i = 0; i < arr.length; i++) {
    		sum += arr[i][col];
    	}
    	return sum;
    }
    
    
    public static double sumColumn(double[][] arr, int col) {
    	double sum = 0;
    	for (int i = 0; i < arr.length; i++) {
    		sum += arr[i][col];
    	}
    	return sum;
    }
    
    
    public static void multiply(double c, double[] arr) {
    	for (int i = 0; i < arr.length;  i++) {
    		arr[i] *= c;
    	}
    }
    
    public static double columnMin(double[][] arr, int col) {
    	double min = Double.POSITIVE_INFINITY;
    	for (int i = 0; i < arr.length; i++) {
    		min = Math.min(arr[i][col], min);
    	}
    	return min;
    }
    
    public static int min(int... nums) {
    	int min = Integer.MAX_VALUE;
    	for (int n : nums) {
    		if (n < min)
    			min = n;
    	}
    	return min;
    	
    }
    
    public static int max(int... nums) {
    	int max = Integer.MIN_VALUE;
    	for (int n : nums) {
    		if (n > max)
    			max = n;
    	}
    	return max;
    	
    }
    
    public static double min(Double... nums) {
    	Double min = Double.POSITIVE_INFINITY;
    	for (Double d : nums) {
    		if (d < min)
    			min = d;
    	}
    	return min;
    	
    }
    
    public static double max(Double... nums) {
    	Double max = Double.NEGATIVE_INFINITY;
    	for (Double d : nums) {
    		if (d > max)
    			max = d;
    	}
    	return max;
    	
    }
    

    
	public static double columnMax(double[][] arr, int col) {
	  	double max = Double.NEGATIVE_INFINITY;
    	for (int i = 0; i < arr.length; i++) {
    		max = Math.max(arr[i][col], max);
    	}
    	return max;
	}
    
    public static Double[] boxArray(double[] arr) {
		Double[] da = new Double[arr.length];
		for (int i = 0; i < arr.length; i++) {
			da[i] = arr[i]; 
		}
		return da;
	}

    public static double[] unboxArray(Double[] arr) {
    	double[] da = new double[arr.length];
		for (int i = 0; i < arr.length; i++) {
			da[i] = arr[i]; 
		}
		return da;
	}
    
    public static void printMatrix(double[][] mat) {
		for (double[] row : mat) {
			System.out.println(Arrays.toString(row));
		}
    }
    
    public static void printMatrix(Double[][] mat) {
		for (Double[] row : mat) {
			System.out.println(Arrays.toString(row));
		}
    }
    
    public static String toString(double[][] mat) {
    	if (mat == null)
    		return "null";
    	String str = "[";
    	int i = 0;
		for (double[] row : mat) {
			str += Arrays.toString(row);
			if (i++ != mat.length - 1) {
				str += ";";
			}
		}
		return str + "]";
    }
    
    public static String toString(Object[][] mat) {
    	String str = "[";
    	int i = 0;
		for (Object[] row : mat) {
			str += Arrays.toString(row);
			if (i++ != mat.length - 1) {
				str += ";";
			}
		}
		return str + "]";
    }

	public static void displayStats(double[] data) {
		double mean = sum(data)/data.length;
		double var = 0;
		for (double d : data) 
			var += (d - mean)*(d - mean);
		var /= data.length;
		System.out.printf("mean = %.4f variance = %.4f\n", mean, var);
	}

	public static int[] sumRows(int[][] mat) {
		int[] sums = new int[mat.length];
		int i = 0;
		for (int[] r : mat) {
			sums[i] = sum(r);
			i++;
		}
		return sums;
	}

	public static int[] sumColumns(int[][] mat) {
		int[] sums = new int[mat[0].length];
		for (int i = 0; i < mat.length; i++) {
			for (int j = 0; j < mat[i].length; j++) {
				sums[j] += mat[i][j];
			}
		}
		return sums;
	}

	public static boolean contains(int val, int[] set) {
		for (int s : set) {
			if (val == s)
				return true;
		}
		return false;
	}
	

	
	/**
	 * 
	 * @param ss
	 * @return
	 */
	public static int[] states(State[] ss) {
		if (ss == null) {
			return null;
		}
		int[] states = new int[ss.length];
		int j = 0;
		for (State st : ss) {
			states[j++] = st.getState(); 
		}
		return states;
	}

    
  
}
