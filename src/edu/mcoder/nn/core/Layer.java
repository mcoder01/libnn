package edu.mcoder.nn.core;

import java.io.Serializable;

public record Layer(int neurons, ActivationFunction activationFunction) implements Serializable {}