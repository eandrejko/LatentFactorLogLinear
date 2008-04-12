/* Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.mahout.matrix;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.WritableComparable;

import java.util.Iterator;

/**
 * Implements subset view of a Vector
 */
public class VectorView extends AbstractVector {
  private Vector vector;

  // the offset into the Vector
  private int offset;

  // the cardinality of the view
  private int cardinality;

  public VectorView(Vector vector, int offset, int cardinality) {
    super();
    this.vector = vector;
    this.offset = offset;
    this.cardinality = cardinality;
  }

  @Override
  protected Matrix matrixLike(int rows, int columns) {
    return ((AbstractVector) vector).matrixLike(rows, columns);
  }

  @Override
  public WritableComparable asWritableComparable() {
    StringBuilder out = new StringBuilder();
    out.append("[");
    for (int i = offset; i < offset + cardinality; i++)
      out.append(getQuick(i)).append(", ");
    out.append("] ");
    return new Text(out.toString());
  }

  @Override
  public int cardinality() {
    return cardinality;
  }

  @Override
  public Vector copy() {
    return new VectorView(vector.copy(), offset, cardinality);
  }

  @Override
  public double getQuick(int index) {
    return vector.getQuick(offset + index);
  }

  @Override
  public Vector like() {
    return vector.like();
  }

  @Override
  public Vector like(int cardinality) {
    return vector.like(cardinality);
  }

  @Override
  public void setQuick(int index, double value) {
    vector.setQuick(offset + index, value);
  }

  @Override
  public int size() {
    return cardinality;
  }

  @Override
  public double[] toArray() {
    double[] result = new double[cardinality];
    for (int i = 0; i < cardinality; i++)
      result[i] = vector.getQuick(offset + i);
    return result;
  }

  @Override
  public Vector viewPart(int offset, int length) throws CardinalityException,
      IndexException {
    if (length > cardinality)
      throw new CardinalityException();
    if (offset < 0 || offset + length > cardinality)
      throw new IndexException();
    Vector result = new VectorView(vector, offset + this.offset, length);
    return result;
  }

  @Override
  public boolean haveSharedCells(Vector other) {
    if (other instanceof VectorView)
      return other == this || vector.haveSharedCells(other);
    else
      return other.haveSharedCells(vector);
  }

  /*
   * (Non-Javadoc)
   * Returns true if index is a valid index in the underlying Vector
   */
  private boolean isInView(int index) { 
    return index>=offset && index<offset+cardinality;
  }

  @Override
  public Iterator<Vector.Element> iterator() { return new ViewIterator(); }
  public class ViewIterator implements Iterator<Vector.Element> {
    Iterator<Vector.Element> it;
    Vector.Element el;
    public ViewIterator() {
      it=vector.iterator();
      while(it.hasNext())
      {	el=it.next();
	if(isInView(el.index())) return;
      }
      el=null;	// No element was found
    }
    public Vector.Element next() { return el; }
    public boolean hasNext() { return el!=null; }
    /** @throws UnsupportedOperationException all the time. method not
     * implemented.
     */
    public void remove() { throw new UnsupportedOperationException(); }
  }
}