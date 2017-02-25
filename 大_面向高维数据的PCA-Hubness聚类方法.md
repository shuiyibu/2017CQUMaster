需要考虑的点：

>- 关于 hubness 自适应的问题
>  - 是否可用谱分析替代 PCA（通过谱分析，找相邻特征值 gap 较大的地方——这个方法我只了解个大概，而且我觉得“较大”这样的词也让它变得不能自动化了。）

# 大_面向高维数据的PCA-Hubness聚类方法

郎江涛

（重庆大学计算机学院，重庆，400044）

# 摘要

​	机器学习（Machine Learning）是一门人工智能的科学，该领域的主要研究对象是人工智能，特别是如何在经验学习中改善具体算法的性能。是通过机器自主学习的方式处理人工智能中的问题。近几十年机器学习在概率论、计算复杂性理论、统计学、逼近论等领域均有发展，已形成一门多领域交叉学科 。机器学习通过设计和分析让机器可以自主“学习”的算法以便从海量数据中自动分析 有价值的模式或规律，从而对未知数据进行预测。机器学习可以大致分为下面四种类别：监督学习（Supervised Learning）、无监督学习（Unsupervised Learning）、半监督学习（Semi-supervised Learning）以及增强学习（Reinforcement Learning）。机器学习已广泛应用于诸多领域：数据挖掘、计算机视觉、搜索引擎、自然语言处理、语音和手写识别、生物特征识别、DNA序列测序、医学诊断、检测信用卡欺诈和证券市场分析等。

​	聚类分析（Cluster Analysis，亦称为群集分析）是把相似的对象通过静态分类的方法分成不同的簇或子集，使得在同一个簇中的对象都具有某些相似的属性。传统的聚类分析计算方法主要有如下五种：划分方法（Partitioning Methods）、层次方法(Hierarchical Methods)、基于密度的方法(Density-Based Methods)、基于网格的方法(Grid-Based Methods)和基于模型的方法(Model-Based Methods)。传统的聚类分析适用于低维数据的聚类问题。然而由于现实世界中数据的复杂性，在使用传统聚类算法处理诸多问题和任务时其表现效果不佳，尤其是对于高维数据和大型或海量数据而言。这是因为在高维数据空间中利用传统算法聚类时会碰到下述两个问题：（1）高维数据存在大量冗余、噪声的特征使得不可能在所有维中均存在簇；（2）高维空间中的数据分布十分稀疏，其数据间的距离几乎相等。显然，基于距离的传统聚类方法无法在高维空间中基于距离来构建簇。这便是机器学习中令人头疼的维数灾难（Curse of Dimensionality）问题[d1]。近年来，“维数灾难”已成为机器学习的一个重要研究方向。与此同时，“维数灾难”通常是用来作为不要处理高维数据的无力借口。随着科技的发展使得数据获取变得愈加容易，而数据规模愈发庞大、复杂性越来越高，如海量Web 文档、基因序列等，其维数从数百到数千不等，甚至更高。高维数据分析虽然十分具有挑战性，但是它在信息安全、金融、市场分析、反恐等领域均有很广泛的应用。

​	为了解决维数灾难的问题，本文引入了 hubness 这一全新的概念。并在原有的基于 hub 的聚类算法进行实验分析后，对基于 hub 的算法进行了改进。Hubness 是在2010 年由 Milosˇ Radovanovic ́ 等人提出的一种全新的概念。在数据集中一个点出现在其它点的 k 近邻列表中的次数称为 k-occurrences，而 hubness 会影响k-occurrences 的分布。随着维度的增加，k-occurrences 的分布会逐渐倾斜，这将会导致 hubs 的出现。Hubs 是指那些具有非常高的 k-occurrences 的点，换言之，hubs 的点易于频繁地出现在其它点的 k 近邻类表中。通过探究这种现象的根源，发现这是高维向量空间数据分布的一种内在属性。同时，对基于距离度量的各种机器学习方法进行了直接或非直接的研究，包括监督学习方法，半监督学习方法和无监督学习方法。针对聚类分析，现有的基于hub的算法有以下4种：deterministic, probabilistic, hybrid 和 kernel。这4种方法均为 k-means 算法的扩展。在 deterministic 方法中，首先确定簇的数量然后使用 k-means 算法进行聚类，在每次聚类的过程中将当前簇中的具有高的 hubness 分数的点作为其中心。Probabilistic 方法使用模拟退火算法以一定概率$\theta(=min(1, t/N Prob))$ 选择高 hubness 分数的点作为当前簇的中心。Deterministic 和 probabilistic 方法只依赖于距离矩阵而不必关心数据的表现形式。为了尽可能地获取数据的中心位置则需要使用 hybrid 方法。在 hybrid 方法中，使用数据点的 hubness 分数来指导搜索，但最终会形成基于质心的簇结构。kernel 方法在前三者基础上可以对非超球面簇集进行处理。基于 hub 的聚类算法用于高维数据，由此可见随着维度的增加聚类时间和迭代次数也随之增加。

​	针对基于 hub 的聚类分析方法的优缺点、适用性等问题，本文提出了一种基于 PCA-Hubness 的聚类分析方法。首先，构建 KNN 邻域矩阵，计算每个点的 k-occurrences 值。然后，用PCA 进行降维，在降维的过程中通过偏度的变化率来控制降维的程度，以防损失过多重要的有价值信息。最后，在获取降维数据后利用基于 hub 的算法进行聚类分析。实验证明，本文提出的基于 PCA-Hubness 的聚类算法可以在基本保持本征维数不变的情况下对数据降维，对给定数据集的聚类效果与传统的 kmeans 算法和基于 hub 的聚类算法相当或者更优。从而很大程度上解决了传统算法无法在高维数据空间中聚类分析的问题。

**关键字：**聚类；高维数据；本征维度；Hubness；PCA

# Abstract



**Keywords**: Clustering; High-dimensional Data; Intrinsic Dimension; Hubness; PCA

> 内容确定后再翻译	

​		
​	

​			

# 1 绪论

​	本章主要介绍论文的选题背景及其意义，阐述论文的研究方向和研究的主要内容，并对论文的整体结构作简要说明。

## 1.1 研究背景及意义

​	当今科学技术发展越来越迅捷，加之云计算等新兴大数据处理技术在计算机诸多领域持续发展，人们对大型数据表现出前所未有的关注。网络信息的快速传播使得现实世界产生的数据几乎呈现出指数增长的趋势。随着网络数据的持续增加和网络数据的结构的持续复杂，使得数据分析变得愈加困难。当今社会数据的过快产生使得我们身在一个“被信息所淹没，但却渴望从中获取知识”的环境中[1]。对于这些大量、增长速度持续增加且结构异常复杂的数据，传统的数据处理方法已变得不再适用。于是，一种基于大数据的处理方法应运而生。数据挖掘的主要目标是从大量数据中提取出有价值的模式和知识，然后将其转变为人类可理解的结构，以便后续的工作使用【2】。

​	在大型的数据集中，数据挖掘通过机器学习、人工智能、统计学等交叉方法从而发现有价值的模式和知识。数据挖掘的过程是对大型数据进行监督或半监督的分析，从而获得之前未知的有意义的潜在信息，例如数据的聚类（通过聚类分析）、数据的异常信息（通过利群点检测）和数据之间的联系（通过关联规则分析）。数据挖掘的对象的类型并无限制，可以使任意类型的数据，不管是结构化的数据、半结构化的数据，还是异构型的数据【3】。数据挖掘的主要过程如下图 1.1 所示。数据挖掘的过程通常定义以下三大阶段：（1）预处理阶段：在获取到目标数据集后，有必要对多变量数据进行分析，处理那些包含噪声和含有缺失数据的观测量；（2）数据挖掘阶段：数据挖掘过程涉及六种常见的任务，异常检测（异常／变化／偏差检测）、关联规则学习（依赖建模）、聚类、分析、回归以及汇总，这些均是利用数据挖掘技术从原有的数据集中发现未知的有价值的信息；（3）结果验证阶段：通常，数据挖掘是有目的地挖掘未知的有价值的信息，然而这些信息是否符合预期一般可以通过结果验证来实现。数据挖掘的方法包括监督式学习、无监督式学习、半监督学习、增强学习。监督学习是从已知的训练数据集中获得某种函数用于预测未知的数据集。监督学习训练集中的目标是由人为标注的。常见的监督式学习包括分类、估计、预测。无监督学习与监督学习的不用之处在于训练集是没有人为标注的。常见的非监督式学习包括聚类、关联规则分析。半监督学习介于监督学习与无监督学习之间。增强学习是基于环境而行动，从而获得最大化的预期利益。

> 上述信息可做简略阐述



> 画流程图

​	‘’物以类聚，人以群分“，无论是自然科学还是现实世界中均有各种各样的分类问题。在数据挖掘中，聚类分析是研究分类问题的一种数据分析方法。聚类分析是把大量复杂的数据通过聚类器将其分成若干不同的类别或更多的子集，换言之，聚类分析的目的是尽可能地增大簇内部的相似性同时减小簇之间的相似性。聚类分析在诸多领域均有应用，包括机器学习、数据挖掘、模式识别、图像分析以及生物信息等。

## 1.2 国内外研究现状

​	随着科学技术的发展，人们处理大型复杂数据的需求也越来越强烈，数据挖掘在学术界一直备受关注。数据挖掘的相关理论不断完善和发展，而且其商业价值也逐渐显著。聚类分析是数据挖掘的重要组成部分，自然也受到了研究者的高度关注。就聚类分析本身而言，它并不是一个具体的算法，而是处理某一类问题的通用规则。不同的聚类器可以定义不同的簇结构以及搜寻它们的规则。主流的簇概念包括簇内对象之间的小距离、数据空间的密集区域以及间隔或特定的分布。因此，聚类分析可以被表示为多目标优化问题。不同的数据集和结果的预期用途决定了聚类算法的选择和参数的设定（包括诸如要使用的距离函数，密度阈值或预期聚类的数量）。聚类分析本身并不是一个自动化的过程，而是一个不断迭代的知识发现过程或是交互式多目标优化的过程。在这个迭代过程中需要不断修改数据预处理方式以及模型参数直到到达预期的结果。聚类分析是在1932年由两位人类学专家 Driver 和 Kroeber 首次提出的，1938年 Zubin 将其引入到了心理学领域。

​	由于难以对簇的概念作出准备定义导致有诸多的聚类算法产生，这些聚类算法使用了不同的聚类模型，主流的聚类模型包括：（1）连通性模型（Connectivity Models），例如层次聚类基于距离连通性构建模型；（2）中心性模型（Centroid Models），例如 k-means 算法将单个平均向量表示每个簇类；（3）分布模型（Distribution Models），使用统计分布对聚类进行建模，例如由 EM 算法使用的是多变量正态分布；（4）密度模型（Density Models），例如 DBSCAN 和 OPTICS 将簇定义为数据空间中的连接密集区域；（5）子空间模型（Subspace Models），在Biclustering（也称为协同聚类或双模式聚类）中使用集群成员和相关属性建模；（6）组模型（Group Models），一些算法不提供精确模型，仅提供分组信息；（7）==基于图的模型（Graph-Based Models）：团体，即图中的节点的子集，使得子集中的每两个节点通过边连接，可以被认为是聚类的原型形式。 完全连通性需求的松弛（边缘的一小部分可能丢失）被称为准丛集，如在HCS聚类算法中（Graph-based models: a clique, that is, a subset of nodes in a graph such that every two nodes in the subset are connected by an edge can be considered as a prototypical form of cluster. Relaxations of the complete connectivity requirement (a fraction of the edges can be missing) are known as quasi-cliques, as in the HCS clustering algorithm.）==。聚类可以细致地分为：（1）严格划分聚类，每个对象正好属于一个簇；（2）包含离群点的严格划分聚类，对象也可以不属于任何簇，那么它将会被视为离群点；（3）重叠聚类（也称作可替代聚类或多视图聚类），虽然通常是硬聚类，但对象也可能属于多个簇；（4）分层聚类：属于子集群的对象同时也属于父集群；（5）==子空间聚类：尽管在唯一定义的子空间内的重叠聚类，聚类不期望重叠（Subspace clustering: while an overlapping clustering, within a uniquely defined subspace, clusters are not expected to overlap.）==。

​	聚类分析算法是根据它们的聚类模型进行分类的，没有客观的“正确的”聚类算法，正如 Vladimir 已经指出的，“聚类是在旁观者的眼中（Clustering is in the eye of the beholder）”。针对特定的问题，除非有数据理论依据，否则需要通过实验进行选择合适的聚类算法 [4]。 下面仅列出最主流的聚类算法。

(1) 层次聚类算法

​	层次聚类算法也称作基于连通性的聚类算法，其核心思想是若两个对象越接近，那么它们的相关性就越强。这些算法基于对象间的距离将彼此连通从而形成不同的簇。在很大程度上，一个簇可以由该簇内的最大连通距离来表示。不同的距离会形成不同的簇，这可以通过树形结构来表示，这也是层次聚类名称的来源。

基于聚类模型可以把聚类算法分为以下几类



​	









 

 

# 参考文献： 

[d1] John，N.，Megatrends：Ten New Directions Transforming Our Lives. NY：Futura，1984：p. 28.

[d2] Data Mining Curriculum. ACM SIGKDD. 2006-04-30.

[d3] 潘有能，XML 挖掘：聚类、分类与信息提取. 杭州：浙江大学出版社，2012

[d4] Estivill-Castro, Vladimir (20 June 2002). "Why so many clustering algorithms — A Position Paper". ACM SIGKDD Explorations Newsletter. 4 (1): 65–75. 





[d1] Richard Ernest Bellman. Dynamic Programmin. Courier Dover Publications. 2003. 

[1] Houle, M. E.，Kriegel, H. P.，Kröger, P.，Schubert,E.，Zimek.A. Scientific and Statistical Database Management[J]，Lecture Notes in ComputerScience **6187**: 482. 2010.

[2] Tony Jebara，Jun Wang，Shih-Fu Chang. Graphconstruction and b-matching for semi-supervised learning[J]. In Proceedings ofthe 26th International Conference on Machine Learning(ICML)， pages441–448，2009.

[3] Amina M，Syed Farook K. A Novel Approach forClustering High-Dimensional Data using Kernel Hubness[J]. InternationalConfenrence on Advances in Computing and Communication. 2015.

[4] Ester Martin，Kriegel Hans-Peter，Sander,Jörg，Xu, Xiaowei，Simoudis Evangelos，Han,Jiawei，FayyadUsama M., eds. A density-based algorithm for discovering clusters in largespatial databases with noise[J]. Proceedings of the Second InternationalConference on Knowledge Discovery and Data Mining (KDD-96). AAAI Press. pp.226–231. 

[5] Milosˇ Radovanovic ́，Alexandros Nanopoulos，MirjanaIvanovic ́. Hubs in Space: Popular Nearest Neighbors in High-DimensionalData[J]，Journal of Machine Learning Research 11 (2010) 2487-2531. 2010

[6] Abdi. H，Williams L.J. Principal componentanalysis[J]. Wiley Interdisciplinary Reviews: Computational Statistics. 2 (4):433–459. 2010

[7] Peter J. Rousseeuw. Silhouettes: a Graphical Aid tothe Interpretation and Validation of Cluster Analysis[J]. Computational andApplied Mathematics. **20**: 53–65.1987.

[8] Nenad Toma sev，Milo s Radovanovi c，DunjaMladeni c，andMirjana Ivanovi c. The Role of Hubness in Clustering High-Dimensional Data[J]，IEEETRANSACTIONS ON KNOWLEDGE AND DATA ENGINEERING, VOL. 26, NO. 3，2014  

 

 

 

 

 

 

 

 