/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.mahout.classifier.sgd;

import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.function.UnaryFunction;
import org.apache.mahout.math.list.IntArrayList;

/**
 * Implements a Latent Factor Log-linear model as described in Dyadic Prediction Using a Latent
 * Feature Log-Linear Model by Aditya K Menon and Charles Elkan. See http://arxiv.org/abs/1006.2156
 * <p/>
 * The way that this works is that the latent factors for left-items is kept in one matrix and
 * latent factors for right-items is kept in another.  Training proceeds by considering each row of
 * left and right weights as if they were weights for a logistic regression or a feature vector.  We
 * then use the right weights as a feature to learn the left weights and vice versa.
 * <p/>
 * Regularization is done using an L1 or L2 scheme to decrease weights on each training step.
 * Nothing fancy is done in terms of per term regularization or learning rate annealing because all
 * updates are dense.
 */
public class LatentLogLinear {
  private final LogLinearModel left, right;
  private double mu0;

  public LatentLogLinear(int factors) {
    left = new LogLinearModel(factors);
    right = new LogLinearModel(factors);
  }

  public void train(int leftId, int rightId, int actual) {
    left.train(leftId, actual, right.weights(rightId));
    right.train(rightId, actual, left.weights(leftId));

    // chase intercept term to the left weights
    left.adjustBias(leftId, right.getBias(rightId));
    right.setBias(rightId, 0);
  }

  public LatentLogLinear learningRate(double mu0) {
    left.learningRate(mu0);
    right.learningRate(mu0);
    return this;
  }

  public LatentLogLinear lambda(double lambda) {
    left.lambda(lambda);
    right.lambda(lambda);
    return this;
  }

  public double getLambda() {
    return left.getLambda();
  }

  private static class LogLinearModel extends AbstractOnlineLogisticRegression {
    private Matrix weights;
    private IntArrayList updates;
    private double mu0;
    private int updateCount;

    private LogLinearModel(int factors) {
      weights = new BlockSparseMatrix(factors + 1);
      updates = new IntArrayList();
    }

    @Override
    public double perTermLearningRate(int j) {
      return 1;
    }

    @Override
    public double currentLearningRate() {
      return mu0 / updateCount;
    }

    public void train(int id, int actual, Vector features) {
      if (updates.size() < id) {
        updates.fillFromToWith(updates.size(), id, 0);
      }
      updates.setQuick(id, updates.getQuick(id) + 1);
      updateCount = updates.getQuick(id);

      beta = weights.viewPart(id, 1, 0, weights.columnSize());
      train(actual, features);
    }

    @Override
    public void regularize(Vector instance) {
      beta.assign(new UnaryFunction() {
        @Override
        public double apply(double arg1) {
          double newValue = arg1 - getLambda() * currentLearningRate() * Math.signum(arg1);
          if (newValue * arg1 < 0) {
            return 0;
          } else {
            return newValue;
          }
        }
      });
    }

    public Vector weights(int id) {
      return weights.getRow(id);
    }

    public void setBias(int id, double value) {
      weights.setQuick(id, 0, value);
    }

    public double getBias(int id) {
      return weights.getQuick(id, 0);
    }

    public void adjustBias(int id, double adjustment) {
      double newValue = getBias(id) * adjustment;
      if (Double.isNaN(newValue) || Double.isInfinite(newValue)) {
        System.out.printf("bad bias\n");
      }
      setBias(id, newValue);
    }

    public void learningRate(double mu0) {
      this.mu0 = mu0;
    }
  }
}
