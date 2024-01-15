package edu.mcoder.nn.core;

import java.util.function.Function;

public enum ActivationFunction {
    ReLU(x -> Math.max(0, x), x -> x <= 0 ? 0d : 1d),
    SIGMOID(x -> 1.0/(1+Math.exp(-x)), x -> x*(1-x));

    private final Function<Double, Double> function, derivative;

    ActivationFunction(Function<Double, Double> function, Function<Double, Double> derivative) {
        this.function = function;
        this.derivative = derivative;
    }

    public Function<Double, Double> getFunction() {
        return function;
    }

    public Function<Double, Double> getDerivative() {
        return derivative;
    }
}
