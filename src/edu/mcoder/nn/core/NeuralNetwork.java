package edu.mcoder.nn.core;

import edu.mcoder.nn.util.Matrix;

import java.io.*;

public class NeuralNetwork implements Serializable {
    private final Layer[] layers;
    private final Matrix[] weights, biases, outputs;

    public NeuralNetwork(int inputs, Layer... layers) {
        this.layers = layers;
        weights = new Matrix[layers.length];
        biases = new Matrix[layers.length];
        for (int i = 0; i < layers.length; i++) {
            weights[i] = Matrix.random(i == 0 ? inputs : layers[i - 1].neurons(),
                    layers[i].neurons(), -1, 1);
            biases[i] = Matrix.random(layers[i].neurons(), 1, -1, 1);
        }

        outputs = new Matrix[layers.length+1];
        for (int i = 0; i < outputs.length; i++)
            outputs[i] = new Matrix(i == 0 ? inputs : layers[i - 1].neurons(), 1);
    }

    public double[] forward(double[] inputs) {
        outputs[0] = Matrix.fromArray(inputs);
        for (int i = 1; i <= layers.length; i++) {
            outputs[i] = weights[i - 1].transpose().dot(outputs[i - 1]).add(biases[i-1]);
            final int j = i-1;
            outputs[i].map(spot -> layers[j].activationFunction().getFunction().apply(spot.getValue()));
        }

        return outputs[layers.length].toArray();
    }

    public void train(double[] inputs, double[] labels, double lr) {
        Matrix output = Matrix.fromArray(forward(inputs));
        Matrix target = Matrix.fromArray(labels);
        Matrix error = Matrix.sub(target, output);

        for (int i = layers.length-1; i >= 0; i--) {
            Layer layer = layers[i];
            Matrix finalError = error;
            Matrix gradients = outputs[i+1].copy().map(spot -> {
                double value = layer.activationFunction().getDerivative().apply(spot.getValue());
                return value*finalError.get(spot.getRow(), 0)*lr;
            });

            weights[i].add(Matrix.dot(outputs[i], gradients.transpose()));
            biases[i].add(gradients);
            error = Matrix.dot(weights[i], error);
        }
    }

    public void save(String path) {
        try(FileOutputStream fos = new FileOutputStream(path);
            ObjectOutputStream oos = new ObjectOutputStream(fos)) {
            oos.writeObject(this);
        } catch(IOException e) {
            System.err.println("Error: unable to save the model!");
        }
    }

    public static NeuralNetwork load(String path) throws IOException, ClassNotFoundException {
        FileInputStream fis = new FileInputStream(path);
        ObjectInputStream ois = new ObjectInputStream(fis);
        return (NeuralNetwork) ois.readObject();
    }
}
