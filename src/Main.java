import edu.mcoder.nn.core.*;
import edu.mcoder.nn.loss.LossUtil;
import edu.mcoder.nn.util.ArrayUtil;

import java.io.IOException;
import java.util.Scanner;

public class Main {
    public static void main(String[] args) {
        NeuralNetwork nn;
        try {
            nn = NeuralNetwork.load("model.mnn");
        } catch (IOException | ClassNotFoundException e) {
            nn = new NeuralNetwork(2,
                    new Layer(4, ActivationFunction.SIGMOID),
                    new Layer(2, ActivationFunction.SIGMOID));

            double[][] xor = {
                    {0, 0},
                    {0, 1},
                    {1, 0},
                    {1, 1}
            };

            double[][] labels = {
                    {1, 0},
                    {0, 1},
                    {0, 1},
                    {1, 0}
            };

            Trainer trainer = new Trainer(nn, 0.01, LossUtil.CROSS_ENTROPY);
            trainer.fit(xor, labels, 1000, 256, true, true);
            nn.save("model.mnn");
        }

        boolean use = true;
        while(use) {
            Scanner input = new Scanner(System.in);
            System.out.print("Insert A: ");
            int a = input.nextInt();
            System.out.print("Insert B: ");
            int b = input.nextInt();
            int output = ArrayUtil.argmax(nn.forward(new double[]{a, b}));
            System.out.printf("XOR(%d, %d)=%d\n", a, b, output);
            System.out.print("Again? (Y/n): ");
            String response = input.next();
            if (response.toLowerCase().startsWith("n"))
                use = false;
            else {
                input.next();
                System.out.println();
            }
        }
    }
}