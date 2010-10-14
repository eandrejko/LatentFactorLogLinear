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

import com.google.common.base.Preconditions;
import com.google.common.collect.Maps;
import org.apache.mahout.math.AbstractMatrix;
import org.apache.mahout.math.AbstractVector;
import org.apache.mahout.math.CardinalityException;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.IndexException;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.MatrixView;
import org.apache.mahout.math.Vector;

import java.util.Iterator;
import java.util.Map;

/**
 * Represents a matrix of extensible number of rows and fixed number of columns. The goal here is to
 * store data nearly as densely as a DenseMatrix, but with the ability to add new rows.  This is
 * done by storing blocks indexed by row/blocksize.
 */
public class BlockSparseMatrix extends AbstractMatrix {
  private int rows = 0;
  private int columns;
  private final Map<Integer, Matrix> data = Maps.newHashMap();
  private final int blockSize = 1;

  public BlockSparseMatrix(int columns) {
    this.columns = columns;
    cardinality[COL] = columns;
  }

  // only for GSON use
  private BlockSparseMatrix() {}

  /**
   * Assign the other vector values to the column of the receiver
   *
   * @param column the int row to assign
   * @param other  a Vector
   * @return the modified receiver
   * @throws org.apache.mahout.math.CardinalityException
   *          if the cardinalities differ
   */
  @Override
  public Matrix assignColumn(int column, Vector other) {
    if (other.size() < rows) {
      throw new CardinalityException(rows, other.size());
    }
    if (other.size() > rows) {
      // extend to correct size
      getRow(other.size() - 1);
    }
    for (Integer blockNumber: data.keySet()) {
      Matrix block = data.get(blockNumber);
      int remainder = rowSize() - blockNumber * blockSize;
      if (remainder < blockSize) {
        int n = Math.min(remainder, blockSize);
        block.viewColumn(column).viewPart(0, n).assign(other.viewPart(blockNumber * blockSize, n));
      } else {
        block.assignColumn(column, other.viewPart(blockNumber * blockSize, blockSize));
      }
    }
    return this;
  }

  /**
   * Assign the other vector values to the row of the receiver
   *
   * @param row   the int row to assign
   * @param other a Vector
   * @return the modified receiver
   * @throws org.apache.mahout.math.CardinalityException
   *          if the cardinalities differ
   */
  @Override
  public Matrix assignRow(int row, Vector other) {
    Preconditions.checkArgument(row >= 0 && row < rows, "Bad row number %d not in [0,%d)", row, rows);
    if (other.size() != columns) {
      throw new CardinalityException(columns, other.size());
    }
    data.get(row / blockSize).assignRow(row % blockSize, other);
    return this;
  }

  /**
   * Return the column at the given index
   *
   * @param column an int column index
   * @return a Vector at the index
   * @throws org.apache.mahout.math.IndexException
   *          if the index is out of bounds
   */
  @Override
  public Vector getColumn(int column) {
    if(column < 0 || column >= columns) {
      throw new IndexException(column, columns);
    }
    return new BlockSparseColumn(this, column);
  }

  /**
   * Return the row at the given index
   *
   * @param row an int row index
   * @return a Vector at the index
   * @throws org.apache.mahout.math.IndexException
   *          if the index is out of bounds
   */
  @Override
  public Vector getRow(int row) {
    if(row < 0) {
      throw new IndexException(row, rows);
    }
    extendToThisRow(row);
    return data.get(row / blockSize).getRow(row % blockSize);
  }

  private void extendToThisRow(int row) {
    Matrix block = data.get(row / blockSize);
    if (block == null) {
      data.put(row / blockSize, new DenseMatrix(blockSize, columns));
    }
    rows = Math.max(row + 1, rows);
    cardinality[ROW] = rows;
  }

  /**
   * Return the value at the given indexes, without checking bounds
   *
   * @param row    an int row index
   * @param column an int column index
   * @return the double at the index
   */
  @Override
  public double getQuick(int row, int column) {
    extendToThisRow(row);
    return data.get(row / blockSize).get(row % blockSize, column);
  }

  /**
   * Return an empty matrix of the same underlying class as the receiver
   *
   * @return a Matrix
   */
  @Override
  public Matrix like() {
    BlockSparseMatrix r = new BlockSparseMatrix(columns);
    // ensure parts are allocated
    r.getRow(this.numRows() - 1);
    return r;
  }

  /**
   * Returns an empty matrix of the same underlying class as the receiver and of the specified
   * size.
   *
   * @param rows    the int number of rows
   * @param columns the int number of columns
   */
  @Override
  public Matrix like(int rows, int columns) {
    BlockSparseMatrix r = new BlockSparseMatrix(columns);
    r.extendToThisRow(rows - 1);
    return r;
  }

  /**
   * Set the value at the given index, without checking bounds
   *
   * @param row    an int row index into the receiver
   * @param column an int column index into the receiver
   * @param value  a double value to set
   */
  @Override
  public void setQuick(int row, int column, double value) {
    extendToThisRow(row);
    rows = Math.max(rows, row + 1);
    cardinality[ROW] = rows;
    data.get(row / blockSize).setQuick(row % blockSize, column, value);
  }

  /**
   * Return the number of values in the recipient
   *
   * @return an int[2] containing [row, column] count
   */
  @Override
  public int[] getNumNondefaultElements() {
    return new int[]{rows, columns};
  }

  /**
   * Return a new matrix containing the subset of the recipient
   *
   * @param offset an int[2] offset into the receiver
   * @param size   the int[2] size of the desired result
   * @return a new Matrix that is a view of the original
   * @throws org.apache.mahout.math.CardinalityException
   *          if the length is greater than the cardinality of the receiver
   * @throws org.apache.mahout.math.IndexException
   *          if the offset is negative or the offset+length is outside of the receiver
   */
  @Override
  public Matrix viewPart(int[] offset, int[] size) {
    if (offset[ROW] >= rows || offset[ROW] < 0) {
      throw new IndexException(offset[ROW], rows);
    }
    if (offset[COL] >= columns || offset[COL] < 0) {
      throw new IndexException(offset[COL], columns);
    }
    if (offset[ROW]+size[ROW] > rows || size[ROW] < 0) {
      throw new IndexException(rows - offset[ROW], size[ROW]);
    }
    if (offset[COL]+size[COL] > columns || size[COL] < 0) {
      throw new IndexException(columns - offset[COL], size[COL]);
    }
    return new MatrixView(this, offset, size);
  }

  private static class BlockSparseColumn extends AbstractVector {
    private final BlockSparseMatrix data;
    private final int column;

    private BlockSparseColumn(BlockSparseMatrix data, int column) {
      super(data.rowSize());
      this.data = data;
      this.column = column;
    }
    /*
     * @param rows    the row cardinality
     * @param columns the column cardinality
     * @return a Matrix
     */

    @Override
    protected Matrix matrixLike(int rows, int columns) {
      return new BlockSparseMatrix(columns);
    }

    /**
     * @return true iff the {@link org.apache.mahout.math.Vector} implementation should be
     *         considered dense -- that it explicitly represents every value
     */
    @Override
    public boolean isDense() {
      return true;
    }

    /**
     * @return true iff {@link org.apache.mahout.math.Vector} should be considered to be iterable in
     *         index order in an efficient way. In particular this implies that {@link #iterator()}
     *         and {@link #iterateNonZero()} return elements in ascending order by index.
     */
    @Override
    public boolean isSequentialAccess() {
      return true;
    }

    /**
     * Iterates over all elements <p/> * NOTE: Implementations may choose to reuse the Element
     * returned for performance reasons, so if you need a copy of it, you should call {@link
     * #getElement} for the given index
     *
     * @return An {@link java.util.Iterator} over all elements
     */
    @Override
    public Iterator<Element> iterator() {
      return new Iterator<Element>() {
        int row = 0;

        @Override
        public boolean hasNext() {
          return row < data.rows;
        }

        @Override
        public Element next() {
          Element r = new Element() {
            int r = row;
            int c = column;

            /**
             * @return the value of this vector element.
             */
            @Override
            public double get() {
              return data.getQuick(r, c);
            }

            /**
             * @return the index of this vector element.
             */
            @Override
            public int index() {
              return c;
            }

            /**
             * @param value Set the current element to value.
             */
            @Override
            public void set(double value) {
              data.setQuick(r, c, value);
            }
          };
          row++;

          return r;
        }

        @Override
        public void remove
          () {
          throw new UnsupportedOperationException("Can't remove from matrix iterator");
        }
      }

        ;
    }

    /**
     * Iterates over all non-zero elements.
     *
     * @return An {@link java.util.Iterator} over all non-zero elements
     */
    @Override
    public Iterator<Element> iterateNonZero() {
      return iterator();
    }

    /**
     * Return the value at the given index, without checking bounds
     *
     * @param index an int index
     * @return the double at the index
     */
    @Override
    public double getQuick(int index) {
      return data.getQuick(index, column);
    }

    /**
     * Return an empty vector of the same underlying class as the receiver
     *
     * @return a Vector
     */
    @Override
    public Vector like() {
      return new DenseVector(data.rowSize());
    }

    /**
     * Set the value at the given index, without checking bounds
     *
     * @param index an int index into the receiver
     * @param value a double value to set
     */
    @Override
    public void setQuick(int index, double value) {
      data.setQuick(index, column, value);
    }

    /**
     * Return the number of values in the recipient
     *
     * @return an int
     */
    @Override
    public int getNumNondefaultElements() {
      return data.rowSize();
    }
  }
}
