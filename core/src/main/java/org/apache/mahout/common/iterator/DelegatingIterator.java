/**
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

package org.apache.mahout.common.iterator;

import java.util.Iterator;

import com.google.common.base.Preconditions;

/**
 * An iterator that delegates to another iterator.
 */
public abstract class DelegatingIterator<T> implements Iterator<T> {
  
  private final Iterator<? extends T> delegate;
  
  protected DelegatingIterator(Iterator<T> delegate) {
    Preconditions.checkArgument(delegate != null, "delegate is null");
    this.delegate = delegate;
  }
  
  @Override
  public final boolean hasNext() {
    return delegate.hasNext();
  }
  
  @Override
  public final T next() {
    return delegate.next();
  }
  
  @Override
  public final void remove() {
    delegate.remove();
  }
  
}
