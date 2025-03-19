---
author: "Anson Wang"
date: "2025-03-18"
description: "Introduction of tree family"
title: "Tree Family"
tags: [
    "Tree Model",
    "Machine Learning"
]
math: true
---

# Introduction

Tree-based methods are widely used in many applications. This article will introduce how they evolved step by step. The simplified implementations can be found [here](https://github.com/ansonuser/micro_trees.git) too. The following models are included:

- [Decision Stump](#decision-stump)
- [Decision Tree](#decision-tree)
- [Random Forest](#random-forest)
- [AdaBoost](#adaboost)
- [Gradient Boosting Trees](#gradient-boost)
- [Other Boosting Models](#other-boosting-models)

## Decision Stump

"Induction of One-Level Decision Trees" was introduced by Iba Wayne et al. It is also called "weaker learners" or "base learners". For each feature, find the best cut to discriminate between two classes.


{{< figure src="/ansonwang/images/one_cut.png" alt="One Cut" caption="Fig1. Example of one cut, if a feature value x > 10 then the model goes to the left leaf and predicts it as blue color. " >}}

### Experiment

Let's practice on iris dataset and see how it works. Select two classes from dataset and random split it to training set and testing set. We can first quantile the feature for faster training. Once training is done, we can check how it performs on testing set.

```bash
ds = DecisionStump()
print("Spit on sepal length")
ds.train(X[:tr_idx, 0], Y[:tr_idx])
result = ds.predict(X[tr_idx:,0])
acc = (result == Y[tr_idx:]).mean()
print("test acc=", acc)
parameter_f1 = ds.get_parameter()[1]

print("="*30)
print("Spit on sepal width")
ds.train(X[:tr_idx, 1], Y[:tr_idx])
result = ds.predict(X[tr_idx:,1])
acc = (result == Y[tr_idx:]).mean()
print("test acc=", acc)
parameter_f2 = ds.get_parameter()[1]

```

{{< figure src="/ansonwang/images/cut2features.png" alt="Predict iris" caption="Fig2. Visualization of decision stumps on iris data set." >}}


Only training on one feature is quite powerful but maybe we can make a better decision by combining all features and go deeper like a "tree".

## Decision Tree

Decision tree select one feature, one cut to optimize defined metric for each split. By recursively doing this process until the stop criterion met.

Several common metric options:
- Information Gain: Measures the reduction in entropy after a split in decision trees. Higher gain means better feature selection.
- Gini Index: Measures impurity in a node. Lower values indicate a purer node, guiding the best split.
- Reduction of Variance: Used in regression trees, it selects splits that minimize output variance in child nodes.

Simple criteria:
- Minimum points on a leaf: Specifies the least number of samples required in a leaf node to prevent overfitting.
- Max depth: Limits tree depth to control complexity and reduce overfitting.

The basic idea is put similar things together as efficient as possible. You can seperate all points by unlimited splits but it won't be robust for unseen data. To prevent overfitting and improve performance, some techniques can be applied, such as postpruning by removing nodes contribute almost nothing and ensemble which will be introduced in random forest's part later on.


### Experiment
Still welcome our old friend, classic iris dataset, but we will use all features and labels this time.

```bash
models = multi_class_trainer(X_tr, Y_tr, DecisionTreeClassifier)
num_classes = len(set(Y))
Y_pred = predictor(X_te, num_classes, models).argmax(axis=1)
print("test accuracy:", (Y_te == Y_pred).mean() )
```
The test accuracy is 0.933 on my notebook, it is already powerful. We can check the tree to see its prediction path:

{{< figure src="/ansonwang/images/decision_path.png" alt="Decision Path" caption="Fig3. Prediction paths of decision tree." >}}


## Random Forest

The probability of most classifiers make same mistake drop exponentially. Use partial features, samples for each tree and ensemble these weak trees to build a strong model. There are different terminoloy like bagging, bootstrap and ExtraTrees all follow the same idea.

### Experiment
Use the same set up in decision tree's experiment 

```bash
rf = Random_Forest(30, 0.7, 1)
rf.train(X_tr, Y_tr)
Y_te_prob = rf.predict_prob(X_te)
print("test accuracy:", (prediction_te.values == Y_te_prob.argmax(axis=1)).mean().item())
```

The test accuracy is 0.98. The ensemble trees elevate the accuracy.

However, the algorithms have been introduced so far don't really learn efficiently. They try to find the best fit in each step but not take feedback in terms of how wrong they are. To introduce the concept, we will see how models can learn from mistakes in the following sections. They might not overpower than previous methods and they work quite will in most of cases.

## AdaBoost

AdaBoost was first proposed by Yoav Freund and Robert Schapire in 1995. The output of multiple weak learners is combined into a weighted sum that represents the final output of the boosted classifier. The revised and improved versions on different tasks were published over these years. We will go deeper in this part.


Let **A** be the final strategy, **L** be the loss function and **n** data points. We target to build **T** trees to minimize the loss function: 
$$ L_A - min_{i=1,...,n} L_i= \sum_{t=1}^T p^tl^t - \sum_{t=1}^T l_i^t$$

where $$ p^tl^t = \sum_{i=1}^np_i^tl_i^t $$ and $$ p_i^t$$ is weight of $$i_{th}$$  data point at time step t, $$l_i^t$$ follows the same idea.

The paper claims: The strategy does not perform "too much worse" than the best strategy i for the sequence.

My think: It is more like minizing the maximum loss within all data points. The maximum loss is also the upper bound of final strategy. 

Two core ideas:
1. Minimize the maximize loss from confession.
2. The combination of weak learners (at least better than random guess) boost. 
 
We will show a model can learn by minizing the upper bound, i.e. $$L_A \leq f(L_i)$$  where f(.) is a function of $$L_i$$.

By thinking how we can minimize the maximize loss, simply put more weight on points the model makes wrong decisions.

Here comes $$\beta$$ strategy:

Let $$A = Hedge(\beta)$$
Given N points and $$w_i$$ is weight of $$i_{th}$$ point.

The weight vector can be written as $$ P^t = \frac{W^t}{\sum_{i=1}^N w_i^t} $$
$$\beta \in [0,1]; i = 1,2,...,N; t=1,2,...,T $$

Update weight of data points $$ w_i^{t+1} = w_i^{t} \beta^{-l_i^t} $$

From t=1 to t=T, we have:

$$w_i^{T+1} = w_i^{1}\beta^{-\sum_{t=1}^T l_i^t}, \ \sum_{t=1}^T l_i^t = L_i$$

By equation (Taylor expansion):

$$\alpha^r \leq 1 - (1-\alpha)r; \ for \ r \in [0,1], \alpha \geq 0  $$

Now, $$ \sum_{i=1}^N w_{i}^{t+1} \leq (\sum_{i=1}^N w_i^t)(1+(1-\beta)p^tl^t) $$

We know $$1+x\leq e^x$$, put all together:

$$ \sum_{i=1}^N ​w_i^{T+1}=\sum_{i=1}^Nw_i^1\beta^{-\sum_{t=1}^Tl_i^t}​\leq \prod_{t=1}^T ​(1+(1-\beta)p^t l^t) \leq exp((1-\beta)\sum_{t=1}^T p^t l^t) $$

=> $$ L_{Hedge(\beta)} \leq \frac{ln(\sum_{i=1}^N w_i^{T+1})}{1-\beta} $$

Let $$S \subseteq \{1, 2,…, N\} $$ and
$$ \sum_{i=1}^n w_i^{T+1} \geq \sum_{i \in S} w_i^{T+1} = \sum_{i \in S} w_i^1  \beta^{-L_i} \geq \beta^{-max_{i \in S} L_i} \sum_{i \in S} w_i^1 $$

Finally, we got an upper bound:

$$ L_{Hedge(\beta)} \leq \frac{-ln(\sum_{i=1}^N w_i^{T+1})}{1-\beta} = \frac{-ln(\sum_{i \in S}w_i^1) + max_{i \in S}L_i ln(\beta)}{1-\beta} $$

Analyze the upper bound:
1. when S takes all data, the first term will be zero. As the total weight goes down, the upper bound is elevated.
2. $$\beta$$ can bee seen as loss of the strategy , the boundary shrink as maximum loss and average loss goes down.

$$\beta_t$$ choose in pratical:
Get error rate $$\epsilon_t$$ first and $$\beta_t = \frac{\epsilon_t}{1-\epsilon_t}$$

Note: domain of $$\beta_t$$ is [0, 1], need to map it into domain if it's not (ex: regresion might be out of [0, 1]).


### Experiment
It's time to meet new friend. In this example, we will see a regression task in diabete dataset from "Trevor Hastie's LARS". Using age, sex, body mass index, average blood pressure, and six blood serum measurements to predict disease progression one year after baseline.

```bash
ada = Adaboost()
loss = defaultdict(list)
iterations = 30
for i in range(iterations):                   
    shuffle_idx = np.arange(N)
    np.random.shuffle(shuffle_idx)
       
    X = X[shuffle_idx]
    Y = Y[shuffle_idx]
    tr_idx = int(len(X)*0.7)
    X_tr, Y_tr = X[:tr_idx, :], Y[:tr_idx]
    X_te, Y_te = X[tr_idx:, :], Y[tr_idx:]
    ada.train(X_tr, Y_tr, T=100, maximum_depth=3, mode="l2")
    regr = AdaBoostRegressor(random_state=i, n_estimators=100)
    regr.fit(X_tr, Y_tr)
    loss["mine"].append(((ada.predict(X_te) - Y_te)**2).mean())
    loss["sklearn"].append(((regr.predict(X_te) - Y_te)**2).mean())
```

Running for 30 times and comparing to adaboost from sklearn:

{{< figure src="/ansonwang/images/adaboost_loss.png" alt="Adaboost Loss" caption="Fig4. The result doesn't differ too much but performance from sklearn seems more stable." >}}

## Gradient Boost

Rather than adjust weight of data points. Gradient-Beased approach uses gradient to estimate residuals. 

Let D  $$\(x_i, y_i\)_{i=1 \dots n}$$ be a dataset and **L** be loss function. 

For the $$m_{th}$$ tree, objective function can be written as:

minimize $$L_m(F_{m-1} + h_m, D)$$, $$h_m$$ is the residual that the $$m_{th}$$ tree needs to fit. 

Consider talyer expansion at $$h_m = 0$$ and take first the partial derivative on $$h_m$$ to solve the equation: $$\frac{\partial{L_m}}{\partial{h_m}}=0 $$, and we got $$ \hat{h_m} = \frac{-L'_m}{L''m}$$

Update the model:

$$F_{m} = F_{m-1} + \hat{h_m}$$

Recursively building tree and updating to the model until the criterion meet.


### Experiment

Here we will see some comparisons in regression task and classification task. 
The default split of sklearn's GDBT Classifier is Friedman-mse. Here I simply use gini-impurity for classification task.  

{{< figure src="/ansonwang/images/gbt_loss.png" alt="GDBT Classifier Loss" caption="Fig5. Loss of iris classification task for 30 runs. " >}}

Friedman-mse :

For two subregions l and r, the improvement can bo formulated as: 


$$ i^2(R_l, R_r) = \frac{w_l w_r}{w_l + w_r}(\bar{y_l} - \bar{y_r})^2$$

For this simple task, let's check how gradient boost and random forest perform.

{{< figure src="/ansonwang/images/rf_gbt_loss.png" alt="Rf vs GDBT Classifier Loss" caption="Fig6. Loss of iris classification task for random forest versue gradient boosting. " >}}


In diabete regression task, the result of comparison between gradient boosting and adaboost is as follows:

{{< figure src="/ansonwang/images/gbr_ada_loss.png" alt="AdaBoost vs GDBT Regressor Loss" caption="Fig7. Loss of diabete regression task for adaboost versus gradient boosting. " >}}

{{< figure src="/ansonwang/images/prediction_ada_gdbt.png" alt="AdaBoost vs GDBT Regressor Prediction" caption="Fig8. Relation of adaboost prediction and gradientboost prediction." >}}


A little insight:
- In simple tasks, fit model with more complex way is not necessary better. 
- Adaboost has more discrete prediction with respect to gradient boosting.


### Other Boosting Models

Gradient methods are quite popular. Many models were built upon this concept. 
**XGBoost** (Extreme Gradient Boosting) propose a weighted quantile to improve the split, efficient handling of missing values and sparse data, optimized I/O for large datasets, and increase scalability by parallelization. **LightGBM**(Light Gradient Boosting Machine) introduces gradient-based one-side sampling which prioritizes instances with larger gradients and exclusive feature bundling which groups mutually features to reduce feature space. **CatBoost** process training instance in time-based order to prevent target and encode categories to numeric value, etc.


## Quick Summary:

Decision stump is one level decision tree. Build a decision tree by multi-level decision stump. Ensemble model reduce the loss by planting many trees to a random forest. For learning from feedback, two methods are introduced, Adaboost learns from put more weights on mistake and Gradient-Based model learns from fitting residual by gradient approach. 





## References
[1] Iba, Wayne, et al."Induction of One-Level Decision Trees." ML92, 1992.

[2] Decision Tree, (wiki)[https://en.wikipedia.org/wiki/Decision_tree]

[3] Ho, Tin Kam. "Random Decision Forests." IEEE, 1998

[4] Random Forest, (wiki)[https://en.wikipedia.org/wiki/Random_forest]

[5] Alan Yuille, Lecture 7. (AdaBoost)[https://www.cs.jhu.edu/~ayuille/courses/Stat161-261-Spring14/LectureNotes7.pdf], 2014

[6] Yoav Freund, et al."A decision-theoretic generalization of on-line  learning and an application to boosting." JCSS, 1997.

[7] Harris Drucker."Improving Regressors using Boosting Techniques." ICML, 1996.

[8] J. Friedman, Stochastic Gradient Boosting, 1999

[9] T. Hastie, et al. "Elements of Statistical Learning Ed. 2 Springer.", 2009.

[10] Chen, Tianqi, et al. "XGBoost: A Scalable Tree Boosting System" ACM, 2016

[11] Cuolin Ke, et al. "LightGBM: A Highly Efficient Gradient Boosting Decision Tree" NIPS, 2017

[12] L Prokhorenkova, et al. "CatBoost: unbiased boosting with categorical features" NIPS, 2017










