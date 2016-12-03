package edu.berkeley.nlp.ling;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Version of Tree anchored to particular spans of a sentence
 * 
 * @author gdurrett
 *
 * @param <L>
 */
public class AnchoredTree<L> implements Serializable {
  private static final long serialVersionUID = 1L;

  final L label;
  final int startIdx;
  final int endIdx;
  final List<AnchoredTree<L>> children;

  public List<AnchoredTree<L>> getChildren() {
    return children;
  }

  public L getLabel() {
    return label;
  }

  public int getStartIdx() {
    return startIdx;
  }

  public int getEndIdx() {
    return endIdx;
  }
  
  public int getSpanLength() {
    return endIdx - startIdx;
  }

  public boolean isLeaf() {
    return getChildren().isEmpty();
  }

  public boolean isPreTerminal() {
    return getChildren().size() == 1 && getChildren().get(0).isLeaf();
  }

  public boolean isPhrasal() {
    return !(isLeaf() || isPreTerminal());
  }

  public List<AnchoredTree<L>> getPreOrderTraversal() {
    ArrayList<AnchoredTree<L>> traversal = new ArrayList<AnchoredTree<L>>();
    traversalHelper(this, traversal, true);
    return traversal;
  }

  public List<AnchoredTree<L>> getPostOrderTraversal() {
    ArrayList<AnchoredTree<L>> traversal = new ArrayList<AnchoredTree<L>>();
    traversalHelper(this, traversal, false);
    return traversal;
  }

  private static <L> void traversalHelper(AnchoredTree<L> tree, List<AnchoredTree<L>> traversal, boolean preOrder) {
    if (preOrder) traversal.add(tree);
    for (AnchoredTree<L> child : tree.getChildren()) {
      traversalHelper(child, traversal, preOrder);
    }
    if (!preOrder) traversal.add(tree);
  }

  public List<AnchoredTree<L>> toSubTreeList() {
    return getPreOrderTraversal();
  }
  
  public static <L> AnchoredTree<L> fromTree(Tree<L> tree) {
    return fromTreeHelper(tree, 0);
  }
  

  /**
   * @param currTree
   * @param offset
   * @return A pair containing the anchored tree produced at this 
   */
  public static <L> AnchoredTree<L> fromTreeHelper(Tree<L> currTree, int offset) {
    if (currTree.isLeaf()) {
      return new AnchoredTree<L>(currTree.label, offset, offset + 1, new ArrayList<AnchoredTree<L>>());
    } else {
      List<AnchoredTree<L>> children = new ArrayList<AnchoredTree<L>>();
      int runningOffset = offset;
      for (Tree<L> child : currTree.getChildren()) {
        AnchoredTree<L> anchoredChild = fromTreeHelper(child, runningOffset);
        runningOffset += anchoredChild.getSpanLength();
        children.add(anchoredChild);
      }
      return new AnchoredTree<L>(currTree.label, offset, runningOffset, children);
    }
  }

  public AnchoredTree(L label, int startIdx, int endIdx, List<AnchoredTree<L>> children) {
    this.label = label;
    this.startIdx = startIdx;
    this.endIdx = endIdx;
    this.children = children;
    if (label == null) {
      @SuppressWarnings("unused")
      int x = 5;
    }
  }

  public AnchoredTree(L label, int startIdx, int endIdx) {
    this(label, startIdx, endIdx, Collections.<AnchoredTree<L>>emptyList());
  }

  public int hashCode() {
    int hc = 1363278 * label.hashCode();
    for (AnchoredTree<L> child : children) {
      hc = (hc * 8236211) + child.hashCode();
    }
    return hc;
  }
}
