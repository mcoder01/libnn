package edu.mcoder.nn.loss;

public interface LossFunction {
    double compute(double[] outputs, double[] labels);
}
