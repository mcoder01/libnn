package edu.mcoder.nn.loss;

import edu.mcoder.nn.util.Matrix;
import edu.mcoder.nn.util.Spot;

public class LossUtil {
    public static LossFunction CROSS_ENTROPY = ((outputs, labels) -> {
        Matrix output = Matrix.fromArray(outputs).normalize();
        Matrix target = Matrix.fromArray(labels).normalize();
        Spot loss = new Spot(0, 0, 0);
        output.forEach(spot -> {
            double qx = spot.getValue();
            double px = target.get(spot.getRow(), spot.getCol());
            loss.setValue(loss.getValue()+px*Math.log(qx));
        });

        return -loss.getValue();
    });

    public static LossFunction MEAN = (((outputs, labels) -> {
        Matrix output = Matrix.fromArray(outputs);
        Matrix target = Matrix.fromArray(labels);
        Matrix error = Matrix.sub(target, output);
        Spot loss = new Spot(0, 0, 0);
        error.forEach(spot -> loss.setValue(loss.getValue()+spot.getValue()));
        return loss.getValue()/error.getRows();
    }));
}
