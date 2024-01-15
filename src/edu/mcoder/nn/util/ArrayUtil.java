package edu.mcoder.nn.util;

public class ArrayUtil {
    public static int argmax(double[] values) {
        int argmax = 0;
        for (int i = 1; i < values.length; i++)
            if (values[i] > values[argmax])
                argmax = i;
        return argmax;
    }
}
