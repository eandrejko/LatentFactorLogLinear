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

import org.apache.mahout.math.CardinalityException;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.MatrixSlice;
import org.apache.mahout.math.MatrixTest;
import org.junit.Test;

/**
 * Created by IntelliJ IDEA. User: tdunning Date: Oct 6, 2010 Time: 4:46:38 PM To change this
 * template use File | Settings | File Templates.
 */
public class BlockSparseMatrixTest extends MatrixTest {
  @Override
  public Matrix matrixFactory(double[][] values) {
    BlockSparseMatrix r = new BlockSparseMatrix(values[0].length);
    int row = 0;
    for (double[] rowValues : values) {
      r.getRow(row).assign(rowValues);
      row++;
    }
    return r;
  }

  @Test
  public void testGetRowIndexOver() {
    // this doesn't fail with BSM's because they extend automagically
    assertEquals(0, test.getRow(5).zSum(), 0);
  }


  @Test
  public void testAssignColumnCardinalityLong() {
    double[] data = {2.1, 3.2, 1, 2, 3, 4, 5};
    test.assignColumn(1, new DenseVector(data));
    assertEquals(7, test.rowSize());
  }

}
