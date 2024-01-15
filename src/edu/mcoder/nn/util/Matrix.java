package edu.mcoder.nn.util;

import java.io.Serializable;
import java.util.function.Consumer;
import java.util.function.Function;

public class Matrix implements Serializable {
    private final Spot[][] spots;
    private final int rows, cols;

    public Matrix(int rows, int cols) {
        this.rows = rows;
        this.cols = cols;

        spots = new Spot[rows][cols];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                spots[i][j] = new Spot(i, j, 0);
    }

    public Matrix transpose() {
        Matrix transposed = new Matrix(cols, rows);
        return transposed.map(spot -> get(spot.getCol(), spot.getRow()));
    }

    public Matrix map(Function<Spot, Double> func) {
        return forEach(spot -> spot.setValue(func.apply(spot)));
    }

    public Matrix forEach(Consumer<Spot> consumer) {
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                consumer.accept(spots[i][j]);
        return this;
    }

    public Matrix add(Matrix m) {
        if (rows != m.rows || cols != m.cols)
            throw new RuntimeException("Error: unable to sum matrices with different sizes!");
        return map(spot -> spot.getValue()+m.get(spot));
    }

    public Matrix sub(Matrix m) {
        if (rows != m.rows || cols != m.cols)
            throw new RuntimeException("Error: unable to subtract a matrix with different sizes!");
        return add(Matrix.multiply(m, -1));
    }

    public Matrix multiply(double v) {
        return map(spot -> spot.getValue()*v);
    }

    public Matrix divide(double v) {
        return multiply(1/v);
    }

    public Matrix dot(Matrix m) {
        if (cols != m.rows)
            throw new RuntimeException("Error: dot multiplication between matrices of incompatible sizes!");

        Matrix result = new Matrix(rows, m.cols);
        return result.map(spot -> {
            double sum = 0;
            for (int i = 0; i < cols; i++)
                sum += get(spot.getRow(), i)*m.get(i, spot.getCol());
            return sum;
        });
    }

    public Matrix norm() {
        Matrix norm = new Matrix(cols, 1);
        return norm.map(spot -> {
            Matrix vector = col(spot.getCol());
            Matrix sqrSum = vector.transpose().dot(vector);
            return Math.sqrt(sqrSum.get(0, 0));
        });
    }

    public Matrix normalize() {
        Matrix norm = norm();
        return map(spot -> spot.getValue()/norm.get(spot.getCol(), 0));
    }

    public Matrix col(int index) {
        Matrix m = new Matrix(rows, 1);
        return m.map(spot -> get(spot.getRow(), index));
    }

    public Matrix row(int index) {
        Matrix m = new Matrix(cols, 1);
        return m.map(spot -> get(index, spot.getRow()));
    }

    public double get(int row, int col) {
        return spots[row][col].getValue();
    }

    public void set(int row, int col, double value) {
        spots[row][col].setValue(value);
    }

    private double get(Spot spot) {
        return get(spot.getRow(), spot.getCol());
    }

    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder("[");
        for (int i = 0; i < rows; i++) {
            if (i > 0) builder.append(" ");
            builder.append("[");
            for (int j = 0; j < cols; j++) {
                builder.append(get(i, j));
                if (j < cols-1)
                    builder.append(", ");
            }
            builder.append("]");
            if (i < rows-1)
                builder.append(",\n");
        }
        builder.append("]");
        return builder.toString();
    }

    public Matrix copy() {
        Matrix copy = new Matrix(rows, cols);
        return copy.map(this::get);
    }

    public double[] toArray() {
        if ((rows > 1 && cols > 1)
            || (rows == 0 || cols == 0))
            throw new RuntimeException("Error: can't convert a " + rows + "x" + cols + " matrix to array!");

        double[] array = new double[Math.max(rows, cols)];
        forEach(spot -> array[Math.max(spot.getRow(), spot.getCol())] = spot.getValue());
        return array;
    }

    public int getRows() {
        return rows;
    }

    public int getCols() {
        return cols;
    }

    public static Matrix add(Matrix m1, Matrix m2) {
        return m1.copy().add(m2);
    }

    public static Matrix sub(Matrix m1, Matrix m2) {
        return m1.copy().sub(m2);
    }

    public static Matrix multiply(Matrix m, double v) {
        return m.copy().multiply(v);
    }

    public static Matrix divide(Matrix m, double v) {
        return m.copy().divide(v);
    }

    public static Matrix dot(Matrix m1, Matrix m2) {
        return m1.copy().dot(m2);
    }

    public static Matrix ones(int rows, int cols) {
        Matrix m = new Matrix(rows, cols);
        return m.map(spot -> 1d);
    }

    public static Matrix random(int rows, int cols, double min, double max) {
        Matrix m = new Matrix(rows, cols);
        return m.map(spot -> Math.random()*(max-min)+min);
    }

    public static Matrix fromArray(double[] array) {
        Matrix m = new Matrix(array.length, 1);
        return m.map(spot -> array[spot.getRow()]);
    }
}
