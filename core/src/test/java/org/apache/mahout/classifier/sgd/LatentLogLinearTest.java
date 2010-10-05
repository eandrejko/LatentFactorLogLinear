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

import com.google.common.collect.Lists;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.function.UnaryFunction;
import org.junit.Test;

import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * Created by IntelliJ IDEA. User: tdunning Date: Oct 4, 2010 Time: 6:10:31 PM To change this
 * template use File | Settings | File Templates.
 */
public class LatentLogLinearTest {
  private static final double[] RETENTION = {0.8, 0.5, 0.25};
  private static final int ITERATIONS = 1;

  @Test
  public void testTrain() {
    int n = 1000;
    Matrix alpha = new DenseMatrix(n, 5);
    Matrix beta = new DenseMatrix(n, 5);

    final Random rand = RandomUtils.getRandom();

    UnaryFunction generator = new UnaryFunction() {
      @Override
      public double apply(double arg1) {
        return rand.nextDouble() * 6 - 3;
      }
    };

    alpha.assign(generator);
    beta.assign(generator);
    beta.set(n - 1, 0, 1);

    List<TestEvent> testData = Lists.newArrayList();
    List<TestEvent> trainingData = Lists.newArrayList();
    LatentLogLinear model = new LatentLogLinear(5);
    for (double retention : RETENTION) {
      for (int left = 0; left < n; left++) {
        for (int right = 0; right < n; right++) {
          double p = logit(alpha.getRow(left).dot(beta.getRow(right)));
          int y = rand.nextDouble() < p ? 1 : 0;

          if (rand.nextDouble() < retention) {
            trainingData.add(new TestEvent(left, right, y));
          } else {
            testData.add(new TestEvent(left, right, y));
          }
        }
      }
    }
    Collections.shuffle(trainingData);
    for (int i = 0; i < ITERATIONS; i++) {
      for (TestEvent event : trainingData) {
        model.train(event.left, event.right, event.y);
      }
    }
  }

  private static class TestEvent {
    int left, right, y;

    private TestEvent(int left, int right, int y) {
      this.left = left;
      this.right = right;
      this.y = y;
    }
  }

  private double logit(double v) {
    return 1 / (1 + Math.exp(-v));
  }
}
