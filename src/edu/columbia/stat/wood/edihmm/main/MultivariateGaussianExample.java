/*
 * Created on Aug 5, 2011
 */
package edu.columbia.stat.wood.edihmm.main;

import java.util.Arrays;
import java.util.List;

import cern.colt.Timer;

import edu.columbia.stat.wood.edihmm.DurationDistribution;
import edu.columbia.stat.wood.edihmm.EDiHMM;
import edu.columbia.stat.wood.edihmm.EmissionDistribution;
import edu.columbia.stat.wood.edihmm.Hyperparameters;
import edu.columbia.stat.wood.edihmm.Sample;
import edu.columbia.stat.wood.edihmm.State;
import edu.columbia.stat.wood.edihmm.distributions.GammaPoissonPair;
import edu.columbia.stat.wood.edihmm.distributions.MultivariateGaussianParams;
import edu.columbia.stat.wood.edihmm.distributions.NormalInverseWishartMultivariateGaussianPair;
import edu.columbia.stat.wood.edihmm.util.Util;
import gov.sandia.cognition.math.matrix.Matrix;
import gov.sandia.cognition.math.matrix.MatrixFactory;
import gov.sandia.cognition.math.matrix.Vector;
import gov.sandia.cognition.math.matrix.VectorFactory;
import gov.sandia.cognition.math.matrix.mtj.DenseMatrixFactoryMTJ;
import gov.sandia.cognition.math.matrix.mtj.DenseVectorFactoryMTJ;

/**
 * An example usage of the ED-iHMM with a 2D Gaussian emissions distribution
 * and Poisson duration distribution. 
 * 
 * @author Jonathan Huggins
 *
 */
public class MultivariateGaussianExample {
	
	static int startStates = 5;
	static int burnin = 50; 
	static int interval = 15;
	static int numSamples = 1; 
	static int seqLength = 200;
	static VectorFactory<? extends Vector> vf = DenseVectorFactoryMTJ.getDenseDefault();
	static MatrixFactory<? extends Matrix> mf = DenseMatrixFactoryMTJ.getDenseDefault();
	
	/**
	 * @param args
	 */
	public static void main(String[] args) {
		// construct an ED-iHMM and sample from it
		Vector[] means = new Vector[] { vf.createVector2D(10, 0), vf.createVector2D(0, 10) };
		Matrix[] covars = new Matrix[] { mf.copyArray(new double[][] { { 1.25, -.4 }, { -.4, .64} }), 
										 mf.copyArray(new double[][] { { 1, 0 }, { 0, 2} }) }; 
		
		MultivariateGaussianParams[] emitParams = new MultivariateGaussianParams[means.length];
		for (int i = 0; i < emitParams.length; i++) {
			emitParams[i] = new MultivariateGaussianParams(means[i], covars[i]);
		}
		Double[] durationParams = new Double[] { 10., 3. };

		NormalInverseWishartMultivariateGaussianPair niwd = new NormalInverseWishartMultivariateGaussianPair(vf.createVector2D(), 1.0, 3, mf.createIdentity(2, 2).scale(10));
		EmissionDistribution<MultivariateGaussianParams, Vector> ed = new EmissionDistribution<MultivariateGaussianParams, Vector>(niwd, emitParams);
		DurationDistribution<Double> dd = new DurationDistribution<Double>(new GammaPoissonPair(1, 100), durationParams);
		System.out.println(ed);
		System.out.println(dd);	
		
		double[] pi0 = new double[] { 1, 0, 0};
		double[][] pi = new double[][] { {  0, 1, 0},
										 { 1,  0, 0} };
		
		Util.printMatrix(pi);

		EDiHMM<MultivariateGaussianParams, Vector, Double> hmm = new EDiHMM<MultivariateGaussianParams, Vector, Double>(ed, dd, pi, pi0);
		
		Hyperparameters hypers = new Hyperparameters();
		hypers.q = .95;
		hypers.temperature = 3;
		
		Vector[] data = new Vector[seqLength];
		hmm.generateData(data, hypers);
		State[] trueSS = hmm.getStateSequence();
		
		System.out.println(trueSS);
		System.out.println(Arrays.toString(data));
		
		// Reinitialize ED-iHMM
		ed = new EmissionDistribution<MultivariateGaussianParams, Vector>(niwd);
		dd = new DurationDistribution<Double>(new GammaPoissonPair(1, 100));
		hmm = new EDiHMM<MultivariateGaussianParams, Vector, Double>(ed, dd);
		
		// Sample from ED-iHMM

		List<Sample<MultivariateGaussianParams, Vector, Double>> samples = hmm.sample(data, hypers, burnin, interval, numSamples, startStates);
		
		System.out.println(Arrays.toString(hmm.getJLL()));
		// Print out results
		int i = 0;
		for (Sample<MultivariateGaussianParams, Vector, Double> s : samples) {
			System.out.println("***Sample " + (++i) + "***");
			Util.printMatrix(s.pi);
			System.out.println("Durations parameters: " + Arrays.toString(s.durationParams));
			System.out.println("Emission paramters: " + Arrays.toString(s.emitParams));
		}
		
	}

}
