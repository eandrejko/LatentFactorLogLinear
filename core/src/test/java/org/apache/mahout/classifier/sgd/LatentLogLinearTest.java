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

import com.google.common.base.Predicates;
import com.google.common.collect.Collections2;
import com.google.common.collect.Lists;
import com.google.common.collect.Sets;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.function.UnaryFunction;
import org.apache.mahout.math.stats.OnlineSummarizer;
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
  private static final int ITERATIONS = 50;
  public static final int FACTORS = 2;

  private final Random rand = RandomUtils.getRandom();

  private UnaryFunction generator = new UnaryFunction() {
    @Override
    public double apply(double arg1) {
      return rand.nextDouble() * 6 - 3;
    }
  };

  @Test
  public void testTrain() {
    int n = 1000;
    Matrix alpha = new DenseMatrix(n, FACTORS);
    Matrix beta = new DenseMatrix(n, FACTORS);

    alpha.assign(generator);
    beta.assign(generator);
    beta.getColumn(0).assign(1);

    List<TestEvent> allData = Lists.newArrayList();
    for (int left = 0; left < n; left++) {
      for (int right = 0; right < n; right++) {
        double p = logit(alpha.getRow(left).dot(beta.getRow(right)));
        int y = rand.nextDouble() < p ? 1 : 0;
        allData.add(new TestEvent(left, right, y, p));
      }
    }
    Collections.shuffle(allData);

    for (double retention : RETENTION) {
      LatentLogLinear model = new LatentLogLinear(FACTORS).learningRate(.1).lambda(1e-8);
      int cut = (int) Math.floor(allData.size() * retention);
      List<TestEvent> testData = allData.subList(cut, allData.size());
      List<TestEvent> trainingData = allData.subList(0, cut);
      for (int i = 0; i < ITERATIONS; i++) {
        Collections.shuffle(trainingData);
        for (TestEvent event : trainingData) {
          model.train(event.left, event.right, event.y);
        }
        OnlineSummarizer ref = new OnlineSummarizer();
        OnlineSummarizer refLL = new OnlineSummarizer();
        OnlineSummarizer actual = new OnlineSummarizer();
        OnlineSummarizer actualLL = new OnlineSummarizer();
        for (TestEvent event : testData) {
          double p = event.p;
          ref.add(Math.min(p, 1 - p));

          refLL.add(event.y * Math.log(p) + (1 - event.y) * Math.log(1 - p));

          double phat = model.classifyScalar(event.left, event.right);
          actual.add(Math.abs(event.y - phat));
          actualLL.add(event.y * Math.log(phat) + (1 - event.y) * Math.log(1 - phat));

        }
        double meanBayesError = ref.getMean();
        double meanBayesLL = refLL.getMean();
        double meanError = actual.getMean();
        double meanLL = actualLL.getMean();
        System.out.printf("%d\t%.2f\t%.3f\t%.3f\t%.3f\t%.3f\n", i, retention, meanBayesError, meanError, meanBayesLL, meanLL);
      }
    }
  }

  private static class TestEvent {
    int left, right, y;
    double p;

    private TestEvent(int left, int right, int y, double p) {
      this.left = left;
      this.right = right;
      this.y = y;
      this.p = p;
    }
  }

  private double logit(double v) {
    return 1 / (1 + Math.exp(-v));
  }
}
