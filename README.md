data exploration

some basic classifiers

try joint label?

- [ ] 实现CRLR  [GitHub - Silver-Shen/Causally-Regularized-Learning: A method which takes advantage of causal features for classification](https://github.com/Silver-Shen/Causally-Regularized-Learning)
	- [ ] 实现并且测试作为baseline
	- [ ] 多分类？
	- [ ] We tuned the parameters in our algorithm and baselines via cross validation by gird searching with validation set.
	- [ ] Y可以是两维的吗？
	- [ ] Sklearn 能否自己定制逻辑斯谛回归损失函数  或者通过Pytorch 直接实现mlp加损失函数？ 

- [ ] DBGR，以及看相应的评价指标
> 另一篇18年的文章我看了一下，他们的思路是在17年助教的那个causal regularized LR上做了改进，首先causal regularized LR在这篇新文章里改名叫做Global Balancing Regression了，它的主要思想并不复杂，就是逻辑斯谛回归的改进，其中它学了一个W作为不同样本的weight，然后对这个W以及逻辑斯谛回归中的beta做正则化
> 然后18年的文章相当于是在Global Balancing Regression上做了一个改进，就是认为输入的feature有点多，噪音大，就加了一个auto encoder降维一下，用降维后的输入来输入进Global Balancing Regression，所以改进其实就是把auto encoder的损失函数也加进Global Balancing Regression里面，又加了一项，但是只能用分步迭代的方式去优化，估计会稍微慢一点，看不到具体的代码实现了
> 我觉得如果我们能首先实现一下第一个（有matlab代码）再实现一下第二个，有一个baseline确实也不错，可以和它比较了。然后借鉴一下它的思想，主要就是对sample level做weighting加上用某些方法做feature selection
- [ ] 其他方法
	- [ ] [GitHub - KunKuang/Differentiated-Confounder-Balancing: The source code of our DCB algorithm in KDD17 paper: Estimating Treatment Effect in the Wild via Differentiated Confounder Balancing](https://github.com/KunKuang/Differentiated-Confounder-Balancing)
	- [ ] [GitHub - ricvolpi/domain-shift-robustness: Code for the paper “Model Vulnerability to Distributional Shifts over Image Transformation Sets”](https://github.com/ricvolpi/domain-shift-robustness)
	- [ ] [GitHub - ricvolpi/generalize-unseen-domains: Code for the paper “Generalizing to Unseen Domains via Adversarial Data Augmentation”, NeurIPS 2018](https://github.com/ricvolpi/generalize-unseen-domains)
- [ ] 混合高斯分布？ 套之前那个MDN模型？ 问斌斌？ 重新调权重！
- [ ] 针对context的情况专门设计
- [ ] 目标函数包含class和context两个 直接输入两个label就行？
- [ ] 普通机器学习模型改损失函数，或者MLP?

- [ ] 不需要预测context
- [ ] 机器学习模型跑一遍，用pipeline加超参数搜索
- [ ] 同一类别的不同context 的样本在空间中距离反而很远，反倒是不同class的同context 的样本在空间中距离近，需要借用其他类别的信息
- [ ] 隔开context和class好吗？
- [ ] 看看去年他们修改了什么
- [ ] Context 和label的层级关系？共享权重  乘起来？或者其他做法
