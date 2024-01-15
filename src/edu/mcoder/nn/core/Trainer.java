package edu.mcoder.nn.core;

import edu.mcoder.nn.loss.LossFunction;
import edu.mcoder.nn.util.ArrayUtil;

public class Trainer {
    private final NeuralNetwork model;
    private final LossFunction loss;
    private final double lr;

    public Trainer(NeuralNetwork model, double lr, LossFunction loss) {
        this.model = model;
        this.loss = loss;
        this.lr = lr;
    }

    public void fit(double[][] data, double[][] labels, int epochs, int batchSize, boolean shuffle, boolean verbose) {
        for (int i = 0; i < epochs; i++) {
            int index = shuffle ? (int) (Math.random()*data.length) : i%data.length;
            double lossSum = 0;
            double accuracySum = 0;
            for (int j = 0; j < batchSize; j++) {
                double[] input = data[index];
                double[] target = labels[index];

                model.train(input, target, lr);
                double[] output = model.forward(input);
                lossSum += loss.compute(output, target);
                if (ArrayUtil.argmax(output) == ArrayUtil.argmax(target))
                    accuracySum++;
            }

            lossSum /= batchSize;
            accuracySum = accuracySum/batchSize*100;
            if (verbose)
                System.out.println("Epoch: " + (i+1) + ", Loss: " + lossSum + ", Accuracy: " + accuracySum + "%");
        }
    }
}
