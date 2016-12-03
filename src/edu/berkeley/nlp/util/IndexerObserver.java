package edu.berkeley.nlp.util;

public interface IndexerObserver<E>
{
	public void handleIndexAdd(E element, int index);

}
