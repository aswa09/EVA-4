## [Code link](https://github.com/aswa09/EVA-4/blob/master/S6/EVA4_S6.ipynb)
**Effect of Regularization on Test Loss and Test Accuracy**

![test_loss_and_test_acc](https://github.com/aswa09/EVA-4/blob/master/S6/test_loss_and_test_acc.png)

From the loss graph we can see that without either L1 or L2 regularization, the losses are relatively low, though starting from a higher region.

L1 on its own seems to push for a relatively higher loss, while L2 though having higher losses than without, has lesser losses than L1.
Combined(L1 and L2), they offer losses in the same region as L1 and L2(separately).

Overall it looks like losses are lesser for L2 than L1

In the Accuracy graph, we can see that, without L1 and L2 GBN has higher accuracy on an average, followed by L2 with GBN and no L1/L2 with BN, but these two don't start strong.

The others fall below the 99% mark

---

**25 Misclassified Images for without L1/L2 with BN**

![without_l1l2_with_bn](https://github.com/aswa09/EVA-4/blob/master/S6/without_l1l2_with_bn.png)
---
**25 Misclassified Images for without L1/L2 with GBN**

![without_l1l2_with_gbn](https://github.com/aswa09/EVA-4/blob/master/S6/without_l1l2_with_gbn.png)
