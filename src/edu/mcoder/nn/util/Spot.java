package edu.mcoder.nn.util;

import java.io.Serializable;

public class Spot implements Serializable {
    private final int row, col;
    private double value;

    public Spot(int row, int col, double value) {
        this.row = row;
        this.col = col;
        this.value = value;
    }

    public int getRow() {
        return row;
    }

    public int getCol() {
        return col;
    }

    public double getValue() {
        return value;
    }

    public void setValue(double value) {
        this.value = value;
    }
}
