---
title: A Brief Survey of Text Mining: Classification, Clustering and Extraction Techniques
---

# 文本挖掘简述：分类，聚类和提取技术

## 摘要

每天生成的文本数量正在迅速增加。这种大量非结构化文本不能被计算机简单处理和感知。因此，需要高效和有效的技术和算法来发现有用的模式。文本挖掘是从文本中提取有意义的信息的任务，近年来得到了很多关注。在本文中，我们描述了几个最基本的文本挖掘任务和技术，包括文本预处理，分类和聚类。此外，我们简要介绍生物医学和医疗保健领域的文本挖掘。

##1引言

由于大量文本数据以各种形式创建，例如社交网络，病历，医疗保险数据，新闻媒体等，Text Mining（TM）领域近年来备受关注。IDC在EMC发起的一份报告中预测，到2020年数据量将增长到40 ZBT[^1]，导致从2010年初开始增长50倍[52]。

[^1]: 1ZB = $10^{21}$bytes

文本数据是非结构化信息的一个很好的例子，它是大多数情况下可以生成的最简单的数据形式之一。非结构化文本很容易被人类处理和感知，但是机器难以理解。不用说，这部分文本是信息和知识的宝贵来源。因此，迫切需要设计方法和算法，以便在各种应用程序中有效地处理这些混乱的文本。

正如下文所述，文本挖掘方法与传统的数据挖掘和知识发现方法有一定的关系。

### 1.1知识发现vs数据挖掘

数据库中的知识发现或知识发现（KDD）和文献中的数据挖掘有各种各样的定义。我们将其定义如下：

数据库中的知识发现是从数据中提取隐含的有效、新的和潜在有用的信息，这是非平凡的[45,48]。 数据挖掘是特定算法从数据中提取模式的应用。KDD旨在发现数据中的隐藏模式和连接。基于上述KDD Bigdas，2017年8月，加拿大哈利法克斯定义KDD是指从数据中发现有用知识的整个过程，而数据挖掘是指此过程中的特定步骤。数据可以像数据库一样结构化，但也可以像简单文本文件中的数据一样非结构化。

数据库中的知识发现是一个过程，涉及将几个步骤应用于感兴趣的数据集以摘录有用信息模式。这些步骤是迭代和交互的，他们可能需要用户做出决定。CROSS工业标准数据挖掘过程（Crisp DM[^2]）模型定义了以下主要步骤：1）了解应用程序和数据，确定KDD过程的目标，2）数据准备和预处理，3）建模，4）评估5）部署。数据清理和预处理是最乏味的步骤之一，因为它需要特殊的方法将文本数据转换为适合数据挖掘算法使用的格式。

[^2]: http://www.crisp-dm.org/ 

数据挖掘和知识发现术语经常互换使用。有些人会认为数据挖掘是知识发现的同义词，即数据挖掘由KDD过程的所有方面组成。第二个定义将数据挖掘视为KDD过程的一部分（参见[45]），并阐述了建模步骤，即选择用于在数据中搜索模式的方法和算法。我们将数据挖掘视为KDD过程的建模阶段。

由于硬件和软件技术的巨大进步，近年来知识发现和数据挖掘方面的研究取得了快速进展。数据挖掘继续从机器学习，数据库，统计和人工智能等不同领域的交叉演变而来，仅举几例，这表明了该领域的潜在跨学科性质。我们简要描述与上述三个研究领域的关系。

数据库对高效分析大量数据至关重要。另一方面，数据挖掘算法可以显着提高分析数据的能力。因此，为了数据完整性和管理考虑，数据分析需要与数据库集成[105]。数据库角度的数据挖掘概述可以在[28]中找到。

机器学习（ML）是人工智能的一个分支，试图定义一套方法来查找数据中的模式，以便能够预测未来数据的模式。机器学习涉及研究可以自动提取信息的方法和算法。数据挖掘中使用了大量的机器学习算法。欲了解更多信息，请参阅[101,126]。

统计学是一门数学科学，它涉及收集，分析，解释或解释以及数据表达[^3]。今天，很多数据挖掘算法都基于统计和概率方法。对于数据挖掘和统计学习有大量的研究[1，62，78，137]。

[^3]: http://en.wikipedia.org/wiki/Statistics 

### 1.2文本挖掘方法

文本挖掘或文本知识发现（KDT）首先由Fledman等人提出，指从文本中提取高质量信息的过程（结构化的如RDBMS数据[28,43]，半结构化的如XML和JSON [39,111,112]，非结构化文本资源如文档，视频和图像）。它广泛涵盖了大量用于分析文本的相关主题和算法，跨越各种社区，包括信息检索，自然语言处理，数据挖掘，机器学习，许多应用领域的网络和生物医学科学。

- 信息检索（Information Retrieval，IR）：信息检索是从满足信息需求的非结构化数据集合中寻找信息资源（通常是文档）的活动[44,93]。 因此，信息检索主要集中在促进信息访问而不是分析信息和查找隐藏模式，而这些是文本挖掘的主要目的。信息检索对文本的处理或转换优先级较低，而文本挖掘可以被视为超越信息访问，进一步帮助用户分析和理解信息并简化决策制定。
- 自然语言处理（NLP）：自然语言处理是计算机科学，人工智能和语言学的子领域，旨在使用计算机来理解自然语言[90,94]。许多文本挖掘算法广泛地利用了NLP技术，如词性标注（POG），句法分析和其他类型的语言分析（参见[80,116]以获取更多信息）。
- 从文本提取信息（IE）：信息提取是从非结构化或半结构化文档自动提取信息或事实的任务[35,122]。它通常作为其他文本挖掘算法的起点。例如，提取实体，名称实体识别（NER）以及它们与文本的关系可以为我们提供有用的语义信息。
- 文本摘要：许多文本挖掘应用程序需要对文本文档进行汇总，以便对主题[67,115]中的大型文档或文档集合进行简要概述。总体上有两类摘要技术：1）摘要汇总，其中摘要包括从原始文本中提取的信息单元。2）相反抽象摘要，其中摘要可能包含原始文档中可能不会出现的“合成”信息（请参阅概述）[6,38]。
- 无监督学习方法：无监督学习方法是试图从未标记数据中找到隐藏结构的技术。他们不需要任何训练阶段，因此可以应用于任何文本数据而无需手动操作。聚类和主题建模是在文本数据中使用的两种常用的无监督学习算法。聚类是将文档集合分割成分区的任务，其中同一组（群集）中的文档与其他聚类中的文档更为相似。在主题建模中，使用概率模型去确定软聚类，其中每个文档在所有聚类上具有概率分布，而不是硬聚类文档。在主题模型中，每个主题都可以表示为词语的概率分布，每个文档表示为主题的概率分布。因此，一个主题类似于一个集群，并且一个主题的文档成员是概率性的[1,133]。
- 监督式学习方法：监督式学习方法是机器学习技术，涉及推断函数或从训练数据中学习分类器，以便对未见数据进行预测。监督方法有很多种，如最近邻分类器，决策树，基于规则的分类器和概率分类器[101,126]。
- 文本挖掘的概率方法：有各种概率技术，包括无监督主题模型，如概率潜在语义分析（pLSA）[66]和潜在狄利克雷分配（LDA）[16]，以及监督学习方法，如条件随机场[85] 可以在文本挖掘的背景下定期使用。
- 文本流和社交媒体挖掘：网上有许多不同的应用程序可以产生大量的文本数据流。新闻流应用程序和路透社和谷歌新闻等聚合器产生大量文本流，为我提供非常宝贵的信息来源。社交网络，特别是Facebook和Twitter不断创建大量文本数据。它们提供了一个平台，使用户可以在广泛的主题中自由表达自己。社交网络的动态性使得文本挖掘的过程很困难，需要特殊的能力来处理差和非标准的语言[58,146]。
- 态度挖掘和情感分析：随着电子商务和网上购物的出现，创造了大量的文字，并且不断增长，关于不同的产品评论或用户意见。通过挖掘这些数据，我们可以找到关于在广告和在线营销中非常重要的主题的重要信息和意见（参见[109]的概述）。
- 生物医学文本挖掘：生物医学文本挖掘是指对生物医学科学领域的文本进行文本挖掘的任务。文本挖掘在生物医学领域的作用有两个方面，它使生物医学研究人员能够有效和高效地从海量数据中获取和提取知识，并通过扩大其他生物医学数据的挖掘来促进和促进生物医学发现 作为基因组序列和蛋白质结构[60]。

## 2文本表示和编码

在大量文档上进行文本挖掘通常是一个复杂的过程，因此为文本建立一个数据结构非常重要，这有助于进一步分析文档[67]。表示文件的最常见方式如*bag of words*（BOW），它考虑每个词（词/短语）的出现次数，但忽略顺序。这种表示导致了一种矢量表示，可以使用机器学习和统计信息中的降维算法进行分析。文本挖掘中使用的三种主要降维技术是潜在语义分析（LSI）[42]，概率潜在语义索引（PLSA）[66]和主题模型[16]。

在许多文本挖掘应用中，特别是信息重新检索（IR），需要对文档进行排序，以便在大型集合上更有效地重新检索[131]。为了能够定义单词在文档中的重要性，文档被表示为矢量，并且数字重要性被分配给每个单词。基于这种思想的三种最常用的模型是向量空间模型（VSM）（参见2.2节），概率模型[93]和推理网络模型[93,138]。

### 2.1文本预处理

预处理是许多文本挖掘算法中的关键组件之一。例如，传统的文本分类框架包括预处理，特征提取，特征选择和分类步骤。尽管确认特征提取[57]，特征选择[47]和分类算法[135]对分类过程有重大影响，预处理阶段可能会对这种成功产生显着影响。Uysal等人[140] 特别在文本分类领域研究了预处理任务的影响。预处理步骤通常由标记化，过滤，词形化和词干等任务组成。在下面我们简要描述它们。

- 标记化：标记化是将字符序列分解成称为标记的单词（单词/短语）的任务，并且可能同时丢弃某些字符（如标点符号）。然后令牌列表被用于进一步处理[143]。
- 过滤：通常在文档上进行过滤以删除一些单词。常用的过滤是停止词的删除。停止词是经常出现在文本中的词，没有太多内容信息（例如介词，连词等）。类似地，文中经常出现的词语有很少的信息来区分不同文档的，并且也很少出现的词语也可能没有显着相关性，可以从文档中移除[119,130]。
- 词形化：词形化是考虑词的形态分析的任务，即将词的各种变形形式组合在一起，以便将它们分析为单个项目。换句话说，词形化方法试图将动词形式映射为无限时和名词为单一形式。为了理解文档，我们首先必须指定文档中每个单词的POS，并且因为POS是单调乏味且容易出错的，实际上，词干方法更好。
- 词干：词干方法旨在获得派生词的词根。词干算法确实是语言的基础。 在[92]中引入的第一种词法算法，但在[110]中发表的词法分析器是英语中使用最广泛的词法分析方法[68]。

### 2.2矢量空间模型

为了允许对算法进行更加正式的描述，我们首先定义一些常用的术语和变量如下：

给定文件集合$D = \{d_1，d_2，...，d_D\}$，令$V = \{w_1，w_2，... ，w_v\}$是集合中的一组不同的单词/术语。V被称为词汇。文档$d∈D$中$w∈V$的频率由$f_d(w)$表示，具有$w$的文档的数量用$f_D(w)$表示。用于文档$d$的术语矢量表示为$\vec{t}_d =(f_d(w_1)，f_d(w_2)，...，f_d(w_v))$。

表示文档的最常用方式是将它们转换为数字向量。这种表示被称为“矢量空间模型”（VSM）。其结构简单并且最初引入了索引和信息检索[121]，VSM广泛应用于各种文本挖掘算法和IR系统，并能够对大量文档进行高效分析[67]。

在VSM中，每个单词由一个变量表示，该变量具有数字值，表示文档中单词的权重（重要性）。有两个主要的权重模型：1）布尔模型：在这个模型中，权重$ω_{ij}> 0$被分配给每个项$w_i∈d_j$。对于任何没有出现在$d_j$中的术语，$ω_{ij} = 0$。2）术语频率 - 逆文件频率（TF-IDF）：最流行的术语加权方案是TF-IDF。设q为该项加权方案，则每个词$w∈d$的权重计算如下：

$q(w) = f_d(w)*log\frac{| D|}{f_d(w)}$,                                      (1)

其中$| D | $是集合D中的文档数量。

在TF-IDF中，术语频率通过逆文档频率IDF来归一化。这种规范化降低了文档集合中更频繁出现的词语的权重。确保文档的匹配更加受到集合中频率相对较低的独特词语的影响。

基于术语加权方案，每个文档由术语权重矢量$ω(d)=(ω(d，w_1)，ω(d，w_2)，...，ω(d，w_v))$表示。我们可以计算两个文档$d_1$和$d_2$之间的相似度。最广泛使用的相似性度量之一是余弦相似度，计算如下：

$s(d_1,d_2) = cos(\theta) = \frac{d_1 \cdot d_2}{\sqrt{\sum_{i=1}^v w_{1i}^2}\cdot \sqrt{\sum_{i=1}^v w_{2i}^2}}$,                          (2)

[120,121]更详细地讨论了术语加权方案和向量空间模型。

## 3分类

文本分类在数据挖掘，数据库，机器学习和信息检索等不同领域得到了广泛的研究，并广泛应用于图像处理，医学诊断，文档组织等各个领域。文本分类旨在将预定义的类分配给文本文档[101]。分类问题定义如下。我们有一个训练集$D = \{d_1，d_2，... ，d_n\}$，使得每个文档$d_i$都被标记为来自集合$L = \{l_1，l_2，...，l_k\}$的标签$l_i$。任务是找到一个分类模型（clas-sifier）$f$:

$f : D \rightarrow  L     \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \                   f(d) = l$,                                          (3)

它可以将正确的类标签分配给新文档$d$（测试实例）。将标签明确分配给测试实例称为硬分类，得出属于标签的概率成为软分类。还有其他类型的分类可以将多个标签[54]分配给测试实例。有关多种分类方法的详细概述，请参见[41，71]。评估各种文本分类算法[147]。许多分类算法已经在不同的软件系统中实现，并且可以公开获得，例如BOW toolkit [98]，Mallet [99]和WEKA[^4]。

[^4]: http://www.cs.waikato.ac.nz/ml/weka/ 

为了评估分类模型的性能，我们随机分配测试集。在用训练集训练分类器之后，我们对测试集进行分类并将估计的标签与真实标签进行比较并测量性能。正确分类的文档与文档总数的比例称为准确度[67]。文本分类的通用评估指标是精确度，召回率和F-1得分。Charu等人[1]定义的指标如下：“精确度是正确实例在确定的正实例中的比例。回想一下所有正面实例中正确实例的百分比。而F-1分数是精确度和召回率的几何平均值。”

$F_1 = 2 \times \frac{precision \times recall}{precision + recall}$,                          (4)

### 3.1朴素贝叶斯分类器

概率分类器近来获得了很多人气，并且有非常好的表现[24,73,84,86,118]。这些概率方法对如何生成数据（文档中的单词）进行假设，并基于这些假设提出概率模型。然后用一组训练样本来估计模型的参数。贝叶斯规则用于对新示例进行分类并选择最有可能生成示例的类[96]。

朴素贝叶斯分类器可能是最简单和使用最广泛的分类器。它使用概率模型对每个类中的文档分布进行建模，假定不同术语的分布彼此独立。尽管这种所谓的“朴素贝叶斯”假设在许多真实世界的应用中显然是错误的，但朴素贝叶斯的表现令人惊讶。

有两种常用于朴素贝叶斯分类的主要模型[96]。两种模型的目标都是根据文档中单词的分布来找出一个类的后验概率。这两种模式的区别在于，一种模式考虑了词语的频率，而另一种模式则没有。

（1）多变量Bernoulli模型：在这个模型中，一个文档由一个表示文档中单词存在与否的二元特征向量表示。因此，词的频率被忽略。 原作可以在[88]中找到。

（2）多项式模型：我们通过将文档表示为*a bag of words*来捕获文档中词的频率。[76,97,101,106]已经引入了多项模型的许多不同变体。 McCallum等人[96]对伯努利和多项模型进行了广泛的比较，并得出结论:

- 如果词汇量很小，伯努利模型可能会胜过多项式模型。
- 对于大量词汇量，多项式模型的表现总是优于伯努利模型，并且如果两种模型的词汇量都选择最佳，那么几乎总是优于伯努利。

这两种模型都假定文档是由$θ$参数化的混合模型生成的。我们使用McCallum el框架[96]，定义如下：

混合模型包含混合成分$c_j∈C =\{c_1，c_2，... ，c_k\}。$ 每个文档$d_i = \{w_1，w_2，... ，w_{n_i}\}$首先根据先验选择一个组件$P(c_j |θ)$，然后使用该组件根据它自己的参数$P(d_i | c_j;θ)$创建文档。 因此，我们可以使用所有混合分量的概率之和来计算文档的可能性：

$P(d_i|c_j;\theta) = \sum_{j=1}^kP(c_j|\theta)P(d_i|c_j;\theta)$                          (5)

我们假设类$L = \{l_1，l_2，...l_k\}$和混合成分之间的一对一对应关系，因此$c_j$表示第j个混合成分和第j个类别。因此，给定一组标记的训练样例，$D = \{d_1，d_2，... ，d_{ | D | }\}$，我们首先学习（估计）概率分类模型的参数θ，然后使用这些参数的估计，我们通过计算每个类$c_j$的后验概率来对测试文档进行分类，给定测试文档 ，并选择最可能的类（最高概率的类）。

$P(c_j|d_i;\hat{\theta}) = \frac{P(c_j|\hat{\theta})P(d_i|c_j;\hat{\theta_j})}{P(d_i|\hat{\theta})} = \frac{P(c_j|\hat{\theta})P(w_1,w_2,...,w_{n_i}|c_j;\hat{\theta_j})}{\sum_{c \in C}P(w_1,w_2,...,w_{n_i}|c;\hat{\theta_c})P(c|\hat{\theta})}$,                   (6)

在基于朴素贝叶斯假设的基础上，文档中的单词彼此独立，因此：

$P(w_1,w_2,...,w_{n_i}|c_j;\hat{\theta_j}) = \prod_{i=1}^{n_i}P(w_i|c_j;\hat{\theta_j})$,                                       (7)

### 3.2最近邻分类器

最近邻分类器是一种基于接近度的分类器，它使用基于距离的度量来执行分类。主要思想是基于类似度量（例如（2.2）中定义的余弦），属于同一类的文档更可能“相似”或彼此接近。测试文档的分类是根据训练集中类似文档的分类标签推断出来的。如果我们考虑训练数据集中的k-最近邻，这种方法被称为k-最近邻分类，并且这些k个邻居中最常见的类被报告为类标签，参见[61,93,117,126 ]了解更多信息和示例。

### 3.3决策树分类器

决策树基本上是训练实例的分层树，其中属性值的条件用于分层分割数据。换句话说，决策树[50]根据每个节点或分支处定义的一组测试递归地将训练数据集划分为更小的子部分。树的每个节点都是对训练实例的某个属性的测试，并且从节点下降的每个分支对应于该属性的值之一。通过从根节点开始分类实例，通过此节点测试属性并向下移动与给定实例中的属性值相对应的树分支。这个过程是递归重复[101]。

在文本数据的情况下，决策树节点上的条件通常根据文本文档中的术语来定义。例如，节点可以依靠文档中特定术语的存在或不存在来细分为其子节点。有关决策树的详细讨论，请参阅[19,41,71,113]。

决策树已经与增强技术结合使用。[49，125]讨论提高技术以提高决策树分类的准确性。

### 3.4支持向量机

支持向量机（SVM）是监督式学习分类算法，在文本分类问题中被广泛使用。SVM是线性分类器的一种形式。文本文档中的线性分类器是使分类决策基于文档特征的线性组合的值的模型。因此，线性预测器的输出被定义为$y = \vec a · \vec x  + b$，其中$\vec x  =（x_1，x_2，...，x_n）$是归一化文档词频矢量，$\vec a  =（a_1 ，a_2，...，a_n）$是系数向量，b是标量。我们可以将分类类标签中的前置词$y =\vec a· \vec x+ b$解释为不同类之间的分离超平面。

SVM最初是在[34,141]中引入的。支持向量机尝试在各个类之间找到一个“好”的线性分隔符[34,142]。单个SVM只能分离两个类，一个正类和一个负类[67]。SVM算法试图从正和负例子中找到具有最大距离ξ（也称为裕度）的超平面。距离超平面距离为ξ的文档称为支持向量，并指定超平面的实际位置。如果两类的文档向量不是线性可分的，则确定超平面，使得文档向量的最少数量位于错误的一侧。

SVM方法的一个优点是它对高维度非常稳健，即学习几乎独立于特征空间的维数。它很少需要特征选择，因为它选择了分类所需的数据点（支持向量）[67]。 文献[74]描述了文本数据是SVM分类的理想选择，这是由于文本的稀疏高维性质以及少量不相关的特征。支持向量机方法已被广泛应用于许多应用领域，如模式识别，人脸检测和垃圾邮件过滤[21,40,108]。关于SVM方法的更深入的理论研究见[75]。

## 4集群

聚类是最流行的数据挖掘算法之一，并且在文本上下文中进行了广泛的研究。它具有广泛的应用，如分类[11,12]，可视化[22]和文档组织[37]。聚类是在一组文档中查找相似文档组的任务。通过使用相似度函数来计算相似度。文本聚类可以处于不同级别的粒度，其中聚类可以是文档，段落，句子或术语。聚类是用于组织文档以增强检索和支持浏览的主要技术之一，例如Cutting等人[36]已经使用聚类来生成大量文档集合的目录。[9]利用聚类来构建基于上下文的检索系统。有关聚类的广泛概述，请参阅[70,82]。有各种各样的软件工具，例如Lemur[^5]和BOW [98]，它们实现了通用的聚类算法。

[^5]: http://www.lemurproject.org/ 

有许多可用于文本数据上下文的聚类算法。文本文档可以表示为二元向量，即考虑文档中是否存在单词。或者我们可以使用更精确的表示法，这些表示法涉及加权法，如TF-IDF（请参阅第2.2节）。

尽管如此，这种朴素的方法通常不适合文本聚类，因为文本数据具有许多不同的特征，这就要求为任务设计文本特定的算法。我们描述了文本表示的一些独特属性：

1. 文本表示具有非常大的维度，但底层数据很稀少。换句话说，绘制文档的词汇量很大（例如$10^5$量级），但是给定的文档可能只有几百个词。 当我们处理诸如推文之类的短数据时，这个问题变得更加严重。
2. 一组给定文档的词汇通常是相互关联的。即数据中的概念的数量远小于特征空间。因此，我们需要设计在聚类任务中考虑词相关性的算法。
3. 由于文档中包含的单词数量各不相同，因此在聚类过程中对文档表示进行规范化处理非常重要。

前面提到的文本特征需要设计专门的表示文本的算法，并在IR社区进行广泛调查。已经提出了很多算法来优化文本表示[120]。

文本聚类算法分为许多不同类型，如凝聚聚类算法，分区算法和概率聚类算法。聚类算法在有效性和效率方面具有多种权衡。对于不同聚类算法的实验比较见[132,151]，对聚类算法的调查见[145]。下面我们介绍一些最常用的文本聚类算法。

### 4.1分层聚类算法

分层聚类算法的名称是因为它们构建了一组可以描述为聚类层次结构的聚类。层次结构可以自上而下（称为分裂）或自下而上（称为凝聚）方式构建。分层聚类算法是基于距离的聚类算法之一，即使用相似度函数来测量文本文档之间的接近程度。在[103,104,144]中可以找到关于文本数据的层次聚类算法的广泛综述。

在自上而下的方法中，我们从包含所有文档的一个集群开始。我们递归地将这个集群分成子集群。在集聚方法中，每个文档最初被视为一个单独的集群。然后，将最相似的群集合并到一起，直到所有文档都包含在一个群集中。聚合算法有三种不同的合并方法：1）单链接聚类：在这种技术中，两组文档之间的相似性是这些组中任何一对文档之间最高的相似度。2）群组平均连锁聚类：在群组平均聚类中，两个群落之间的相似度是这些群组中文档对之间的平均相似度。3）完整链接聚类：在这种方法中，两个聚类之间的相似度是这些组中任何一对文档之间最差的相似度。有关这些合并技术的更多信息，请参见[1]。

### 4.2 k-means聚类

k-means聚类是数据挖掘中广泛使用的分区算法之一。k均值聚类将文本数据上下文中的n个文档分为k个聚类。代表群集所围绕的代表性。k-means算法的基本形式是：

为k-均值聚类找到一个最优解是计算困难（NP-hard），然而，有效的启发式算法[18]被用来快速收敛到局部最优。k-均值聚类的主要缺点是它对k的最初选择确实非常敏感。因此，有一些技术用于确定初始k，例如使用另一种轻量级聚类算法，如凝聚聚类算法。更高效的k均值聚类算法可以在[7,79]中找到。

![](https://i.loli.net/2018/05/04/5aec4c5d419f6.png)

### 4.3概率聚类和主题模型

主题建模是最近流行的概率聚类算法之一，近来越来越受到人们的关注。主题建模的主要思想[16,55,66]是为文本文档的语料库创建一个概率生成模型。在主题模型中，文档是主题的混合体，其中主题是对单词的概率分布。

两个主要模型是概率潜在语义分析（pLSA）[66]和潜在狄利克雷分配（LDA）[16]。Hofmann（1999）介绍了pLSA的文件建模。pLSA模型在文档级别没有提供任何概率模型，这使得难以概括模型来模拟新的看不见的文档。Blei等人[16]通过在每个文档的主题混合权重之前引入Dirichlet来扩展该模型，并称为模型潜在狄利克雷分配（LDA）。在本节中，我们将描述LDA方法。

潜在Dirichlet分配模型是用于提取文档集合的专题信息（主题）的最先进的非监督技术[16，56]。 其基本思想是将文档表示为潜在主题的随机混合，其中每个主题是对单词的概率分布。LDA图形表示如图1所示。

![](https://i.loli.net/2018/05/04/5aec52bc30c52.png)

令$D = \{d_1，d_2，... ，d_{ | D |} \}$为语料库，$V = \{w_1，w_2，... ，w_{ | V | } \}$是语料库的词汇表。主题$z_j，1≤j≤K$被表示为| V |上的多项概率分布,$p(w_i|z_j),\sum_i^{|V|}p(w_i|z_j) = 1$.LDA通过两阶段过程生成单词：单词从主题生成，主题由文档生成。 更正式地说，给出文件的文字分布计算如下：

$p(w_i|d) = \sum_{j=1}^Kp(w_i|z_j)p(z_j|d)$                 (8)

LDA假设语料库D的生成过程如下：

1. 对于每个主题$k∈\{1,2，...，K\}$，采样词分布$φ_k〜Dir（β）$

2. 对于每个文档$d∈\{1，2，...，D\}，$

   a. 采样主题分布$θ_d〜Dir（α）$

   b. 对于每个单词$w_n$，其中$n∈\{1,2，...，N\}$，在文件d中，

   -  采样主题$z_i〜Mult（θ_d）$
   - 抽样词$w_n〜Mult（φ_{zi}）$

模型（隐藏和观测变量）的联合分布是：

$P(\phi_{1:K},\theta_{1:D},z_{1:D},w_{1:D}) = \prod_{j=1}^KP(\phi_j|\beta)\prod_{d=1}^DP(\theta_d|\alpha) \times (\prod_{n=1}^NP(z_{d,n}|\theta_d)P(w_{d,n}|\phi_{1:K},z_{d,n}))$,              (9)

#### 4.3.1 LDA的推断和参数估计

给定观察文件，我们现在需要计算隐藏变量的后验分布。因此，下一步：

$P(\varphi_{1:K},\theta_{1:D},z_{1:D}|w_{1:D}) = \frac{P(\varphi_{1:K},\theta_{1:D},z_{1:D},w_{1:D})}{P(w_{1:D})}$,                           (10)

由于分母（在任何主题模型下观察观察的语料库的概率），该分布难以计算[16]。

尽管后验分布（精确推理）不易处理，但可以使用各种各样的近似推理技术，包括变分推理[16]和吉布斯采样[56]。吉布斯抽样是一种马尔可夫链蒙特卡罗算法[53]，试图从后验收集样本以经验分布来近似。

吉布斯抽样计算每个单词后面的主题分配如下：

$P(z_i=k|w_i=w,z_{-i},w_{-i},\alpha,\beta) = \frac{n_{k,-i}^{(d)}+\alpha}{\sum_{k^`= 1}^Kn_{k^`,-i}^{(d)}+K\alpha} \times \frac{n_{w,-i}^{(k)}+\beta}{\sum_{w^`= 1}^Wn_{w^`,-i}^{(k)}+W\beta}$,                        (11)

其中$z_i = k$是单词i到主题k的主题分配，$z_{-i}$是指所有其他单词的主题分配。$n_{w,-i}^{(k)}$是分配给话题k的词w的次数，不包括当前分配。同样，$n_{k,-i}^{(d)}$是主题k分配给文档d中除了当前分配以外的任何单词的次数。有关吉布斯采样的理论概述，请参阅[23,64]。

可以将LDA作为更复杂模型中的模块轻松用于更复杂的目标。此外，LDA已广泛用于各种领域。[27]将LDA与概念层次结合起来，对文档进行建模。[2，5]分别基于LDA开发了基于本体的主题模型，用于自动主题标注和语义标注。文献[4]提出了一种基于知识的情境感知推荐主题模型[81,127]，其基于LDA定义了更复杂的主题模型，用于实体消歧，[3]和[63]提出了一个实体 - 主题模型分别发现连贯主题和实体链接。此外，已经创建了许多LDA变体，例如监督LDA（sLDA）[15]，分层LDA（hLDA）[14]和分层弹球分配模型[HPAM][100]。

## 5信息提取

信息提取（IE）是从非结构化或半结构化文本中自动提取结构化信息的任务。换句话说，信息提取可以被认为是完全自然语言理解的一种有限形式，我们正在寻找的信息事先已知[67]。例如，请考虑以下句子：“微软由比尔盖茨和保罗艾伦于1975年4月4日创建。”

我们可以识别以下信息：

> FounderOf(Bill Gates, Microsoft) 
>
> FounderOf(Paul Allen, Microsoft) 
>
> FoundedIn(Microsoft, April - 4 1975)

IE是文本挖掘中的关键任务之一，在信息检索，自然语言处理和Web挖掘等不同研究领域得到广泛研究。同样，它在生物医学文本挖掘和商业智能等领域有着广泛的应用。有关信息提取的一些应用，参见[1]。

信息提取包括两个基本任务，即实体识别和关系提取。两项任务中的最新技术状态都是统计学习方法。在下面我们简要介绍两个信息提取任务。

### 5.1命名实体识别（NER）

命名实体是标识一些真实世界实体的单词序列，例如， “谷歌公司”，“美国”，“巴拉克奥巴马”。命名实体识别的任务是定位自由文本中的命名实体并将其分类为预定义的类别，如人员，组织，位置等。NER不能通过简单地对字典进行字符串匹配来完成，因为a）词典通常不完整，不包含给定实体类型的所有形式的命名实体。b）命名实体经常依赖于上下文，例如“大苹果”可能是水果或纽约的昵称。

命名实体识别是关系提取任务中的预处理步骤，也有其他应用，如问题回答[1,89]。大多数命名实体识别技术是统计学习方法，如隐马尔可夫模型[13]，最大熵模型[29]，支持向量机[69]和条件随机域[128]。

### 5.2隐马尔可夫模型

标准概率分类技术通常不会考虑相邻词的预测标签。考虑到这一点的概率模型是隐马尔可夫模型（HMM）。隐马尔可夫模型假设马尔可夫过程，其中标签或观察的生成取决于一个或几个先前的标签或观察值。因此，对于观测序列$X =（x_1，x_2，...，x_n）$给出一系列的labers$Y =（y_1，y_2，...，y_n）$，我们有：

$y_i \sim p(y_i|y_{i-1}),x_i \sim p(x_i|x_{i-1})$,                              (12)

隐马尔可夫模型已成功用于命名实体识别任务和语音识别系统。有关隐马尔可夫模型的概述，请参见[114]。

### 5.3条件随机场

条件随机场（CRF）是序列标记的概率模型。由Lafferty等人首先引入[85]。我们参考[85]中有关观测（待标记数据序列）和Y（标记序列）的条件随机场的相同定义，如下所示：

定义：假设$G =（V，E）$是一个使得$Y =（Y_v）v∈V$的图，所以Y由G的顶点索引。那么（X，Y）是一个条件随机场，当随机变量$Y_v$， 以X为条件，服从关于图的马尔可夫性质，并且：

$p(Y_v|X,Y_w,w \neq v) = p(Y_v|X,Y_w,w \sim v)$,                (13)

其中w〜v表示w和v在G中相邻。

条件随机场广泛用于信息提取和词性标注[85]

### 5.4关系提取

关系抽取是另一项基本的信息抽取任务，是查找和定位文本文档中实体之间语义关系的任务。有许多不同的技术提议用于关系提取。最常见的方法是将任务视为分类问题：给定一个句子中同时存在的几个实体，如何将两个实体之间的关系分类为固定关系类型之一。关系可能跨越多个句子，但这种情况很少见，因此，大部分现有的工作都集中在句子中的关系提取。许多使用分类方法进行关系提取的研究已经完成，例如[25,26,59,72,77]。

##6生物医学和医疗保健的生物医学本体和文本挖掘

生物医学科学是文本挖掘被广泛使用的领域之一。生物医学文献呈指数增长，Cohen和Hunter [31]表明PubMed / MEDLINE出版物的增长是惊人的，这使得生物医学科学家难以吸收新的出版物并跟上他们自己研究领域的相关出版物。

为了克服这种文本信息过载并将文本转换为机器可理解的知识，需要自动文本处理方法。因此，文本挖掘技术以及统计机器学习算法被广泛用于生物医学领域。文本挖掘方法已被用于各种生物医学领域，例如蛋白质结构预测，基因聚类，生物医学假说和临床诊断等等。在本节中，我们简要介绍生物医学领域的一些相关研究，包括生物医学本体，然后着手解释生物医学学科中的一些文本挖掘技术，用于命名实体识别和关系提取的基本任务。

### 6.1生物医学本体论

我们首先定义本体论的概念。我们使用W3C的OWL用例和需求文档[^6]中提出的定义如下：

本体正式定义了用于描述和表示域的一组通用术语。本体定义了用于描述和表示知识领域的术语。

根据上面的定义，我们应该提及关于本体的几点：1）本体是领域特定的，即它用于描述和表示知识领域，如教育领域，医学等[149]。2）本体论由这些术语之间的术语和关系组成。术语通常称为类或概念，关系称为属性。

有很多生物医学本体论。关于生物医学本体论的综合列表，请参阅Open Biomedical Ontologies（OBO）[^7]和国家生物医学本体中心（NCBO）[^8]。NCBO本体可通过BioPortal[^9]访问和共享。在下文中，我们简要描述一个生物医学领域最广泛使用的本体论：

- 统一医学语言系统（UMLS）：UMLS[^10][95]是最全面的知识资源，统一了Metathesaurus（大量词汇，其数据来自各种生物医学叙词表）的100多种词典，术语和本体，这些词汇由国家图书馆设计和维护 医学（NLM）。它提供了一种整合所有主要生物医学词汇的机制，例如MeSH，系统化医学临床术语（SNOMED CT），基因本体论（GO）等。

[^6]: http://www.w3.org/TR/webont-req/
[^7]: http://www.obofoundry.org/
[^8]: http://www.bioontology.org/
[^9]: http://bioportal.bioontology.org/
[^10]: https://uts.nlm.nih.gov/home.html

它还提供了一个语义网络，解释Metathesaurus条目之间的关系，即包含关于生物医学术语和普通英语单词的词汇信息的字典以及一组词汇工具。语义网络包含语义类型和语义关系。语义类型是Metathesaurus条目的类别（概念），语义关系是语义类型之间的关系。有关UMLS的更多信息，请参见[17，148]。

除了上述本体和知识来源之外，还有各种本体论更专注于生物辩证子领域。例如，药物基因组学知识库[^11]包括临床信息，包括给药指导线和药物标签，潜在临床可行的基因药物关联和基因型 - 表型关系。

早先描述的本体和知识库广泛用于不同的文本挖掘技术，如生物医学领域的信息提取和聚类。

[^11]: http://www.pharmgkb.org/ 

### 6.2信息提取

如前所述（第5节），信息提取是以自动方式从非结构化文本中提取结构化信息的任务。在生物医学领域，非结构化文本主要包括生物医学文献中的科学论文和临床信息系统中的临床信息。信息提取通常被认为是其他生物医学文本挖掘应用中的预处理步骤，如问题回答[10]，知识提取[124,136]，假设生成[30,91]和摘要[65]。

#### 6.2.1命名实体识别（NER）。 

命名实体识别是信息提取的任务，用于将生物医学实体定位和分类，如蛋白名称，基因名称或疾病[87]。 可以利用本体来为提取的实体提供语义的，明确的表示。 NER在生物医学领域颇具挑战性，因为：

1. 生物医学领域中存在大量与语义相关的实体，并且随着这一领域的新发现而迅速增加。实体数量的这种不停增长对于NER系统是有问题的，因为它们依赖于由于科学文献的不断进步而永远不能完成的词典字典。
2. 在生物医学领域，相同的概念可能有许多不同的名称（同义词）。例如，“心脏病发作”和“心肌梗塞”指向相同的概念，并且NER系统应该能够识别相同的概念，不管表达方式是否不同。
3. 使用缩略语和缩写在生物医学文献中很常见，这使得识别这些术语所表达的概念变得很复杂。

在分析大量文本时，NER系统具有高质量和表现良好至关重要。精确度，召回率和F分数是NER系统中使用的典型评估方法。尽管可靠地获得和比较评估方法存在一些挑战，例如， 如何界定正确识别的实体的边界，NER系统已经证明总体上取得了良好的结果。

NER方法通常分为几种不同的方法：

- 基于字典的方法是主要的生物医学NER方法之一，它使用生物医学术语的详尽词典来定位文本中的实体提及。它决定文本中的单词或短语是否与字典中的某个生物医学实体相匹配。基于辞典的方法主要用于更先进的NER系统。
- 基于规则的方法，定义规定生物医学实体模式的规则。 Gaizauskas等人[51]使用了无文字的语法来识别蛋白质结构。
- 统计学方法，基本上使用一些机器学习方法，通常是监督或半监督算法[139]来识别生物医学实体。 统计方法通常分为两个不同的组别：
  - 基于分类的方法，将NER任务转换为适用于单词或短语的分类问题。朴素贝叶斯[107]和支持向量机[83,102,134]是用于生物医学NER任务的常见分类器。
  - 基于序列的方法，使用完整的单词序列而不是单个单词或短语。他们尝试在训练集上训练之后预测一系列单词的最可能标签。 隐马尔可夫模型（HMM）[32,129,150]，最大熵马尔可夫模型[33]和条件随机场（CRF）[128]是最常见的基于序列的方法，CRF经常被证明是更好的统计生物医学NER系统。
  - 混合方法依赖于多种方法，如将字典或基于规则的技术与统计方法相结合。 [123]引入了一种混合方法，在这种方法中，他们使用基于字典的方法来搜索已知的蛋白质名称以及词性标注（基于CRF的方法）。

#### 6.2.2关系提取

生物医学领域中的关系提取涉及确定生物医学实体之间的关系。给定两个实体，我们的目标是定位它们之间特定关系类型的发生。实体之间的关联通常是二元的，但是，它可以包含两个以上的实体。例如，在基因组领域，重点主要是提取基因和蛋白质之间的相互作用，如蛋白质 - 蛋白质或基因 - 疾病关系。关系提取遇到了与NER相似的挑战，例如创建高质量的注释数据用于培训和评估关系提取系统的性能。欲了解更多信息，请参阅[8]。

生物医学关系提取有许多不同的方法。最直接的技术是基于实体共同发生的地方。如果他们经常一起提到，他们很有可能以某种方式相关。但我们无法仅使用统计数据来识别关系的类型和方向。共现方法通常会导致高召回率和低精度。

基于规则的方法是用于生物医学关系提取的另一种方法。规则可以由领域专家手动定义，也可以通过使用注释语料库中的机器学习技术自动获得。基于分类的方法也是生物医学领域中关系提取的常用方法。使用有监督的机器学习算法完成了一项伟大的工作，该算法能够检测和发现各种类型的关系，例如[20]，他们在PubMed摘要中提取的疾病与治疗之间的关系以及GeneRIF数据库中的基因与疾病之间的关系。

### 6.3概括

概括介绍了大量利用信息提取任务的常见生物医学文本挖掘任务之一。总结是识别一个或多个文档的重要方面并自动以一致的方式表示它们的任务。近年来，由于科技文章和临床信息等生物医学领域非结构化信息的巨大增长，它得到了很大的关注。

生物医学摘要通常是面向应用程序的，可能会用于不同的目的。基于其目的，可以创建各种文档摘要，诸如针对单个文档的内容的单文档摘要和考虑多个文档的信息内容的多文档摘要。

摘要方法的评估在生物医学领域中确实是具有挑战性的。因为决定摘要是否“好”往往是主观的，而且对摘要进行人工评估也很难执行。有一种流行的总结性自动评估技术，称为ROUGE（Giving评估的召回导向研究）。ROUGE通过将自动生成的摘要与人类创建的理想摘要进行比较来衡量自动生成的摘要的质量。该度量通过计算计算机生成的摘要和理想的人工生成的摘要之间的重叠词来计算。有关各种生物医学摘要技术的综合概述，请参见[1]。

### 6.4问题回答

问答是另一个生物医学文本挖掘任务，它显着利用信息提取方法。问答被定义为用自然语言对人类提出的问题做出准确答案的过程。由于数据超载和该领域的信息不断增长，问答在生物医学领域非常关键。

为了产生精确的回答，问题回答系统广泛使用自然语言处理技术。问答系统的主要步骤如下：a）系统接收自然语言文本作为输入。b）使用语言分析和问题分类算法，系统确定提出问题的类型和它应该产生的答案。c）然后它生成一个查询并将其传递到文档处理阶段。d）在文档处理阶段，系统提供查询，系统将查询发送到搜索引擎，取回检索到的文档并提取文本的相关片段作为候选答案，并将它们发送到应答处理阶段。e）回答处理阶段，分析候选答案，并根据它们与问题处理步骤中建立的预期答案类型相匹配的程度进行排列。f）选择排名最高的答案作为问答系统的输出。

生物医学学科中的问答系统已经开始在整个处理步骤中利用和纳入语义知识，以创建更准确的响应。这些基于生物医学语义知识的系统使用各种语义组件，例如在知识源和本体中表示的语义元数据以及语义关系来产生信息。 关于不同的生物医学问答技术的完整概述，请参见[1,10]。

## 7讨论

在本文中，我们试图简要介绍文本挖掘领域。我们概述了在文本领域广泛使用的一些最基本的算法和技术。本文还概述了生物医学领域中的一些重要的文本挖掘方法。尽管对于本文的局限性来说，不可能全面地描述所有不同的方法和算法，但应该对文本挖掘领域当前的进展做一个大致的概述。

考虑到每年生产的科学文献数量很大，文本挖掘对科学研究至关重要[60]。这些在线科学文章的大型档案正在显着增长，因为每天都会增加大量新文章。虽然这种增长使研究人员能够轻松获取更多科学信息，但也使得他们很难找到更符合其兴趣的文章。因此，处理和挖掘大量文本是研究人员非常感兴趣的。

## 参考文献

[1]	Charu C Aggarwal and ChengXiang Zhai. 2012. Mining text data. Springer.

[2]	Mehdi Allahyari and Krys Kochut. 2015. Automatic topic labeling using ontology-based topic models. In Machine Learning and Applications (ICMLA), 2015 IEEE 14th International Conference on. IEEE, 259–264.

[3]	Mehdi Allahyari and Krys Kochut. 2016. Discovering Coherent Topics with Entity Topic Models. In Web Intelligence (WI), 2016 IEEE/WIC/ACM International Conference on. IEEE, 26–33.

[4]	Mehdi Allahyari and Krys Kochut. 2016. Semantic Context-Aware Recommenda-tion via Topic Models Leveraging Linked Open Data. In International Conference on Web Information Systems Engineering. Springer, 263–277.

[5]	Mehdi Allahyari and Krys Kochut. 2016. Semantic Tagging Using Topic Models Exploiting Wikipedia Category Network. In Semantic Computing (ICSC), 2016 IEEE Tenth International Conference on. IEEE, 63–70.

[6]	M. Allahyari, S. Pouriyeh, M. Assefi, S. Safaei, E. D. Trippe, J. B. Gutierrez, and K. Kochut. 2017. Text Summarization Techniques: A Brief Survey. ArXiv e-prints (2017). arXiv:1707.02268

[7]	Khaled Alsabti, Sanjay Ranka, and Vineet Singh. 1997. An efficient k-means clustering algorithm. (1997).

[8]	Sophia Ananiadou, Sampo Pyysalo, Jun’ichi Tsujii, and Douglas B Kell. 2010. Event extraction for systems biology by text mining the literature. Trends in biotechnology 28, 7 (2010), 381–390.

[9]	Peter G Anick and Shivakumar Vaithyanathan. 1997. Exploiting clustering and phrases for context-based information retrieval. In ACM SIGIR Forum, Vol. 31. ACM, 314–323.

[10]	Sofia J Athenikos and Hyoil Han. 2010. Biomedical question answering: A survey. Computer methods and programs in biomedicine 99, 1 (2010), 1–24.

[11]	L Douglas Baker and Andrew Kachites McCallum. 1998. Distributional clustering of words for text classification. In Proceedings of the 21st annual international ACM SIGIR conference on Research and development in information retrieval. ACM, 96–103.

[12]	Ron Bekkerman, Ran El-Yaniv, Naftali Tishby, and Yoad Winter. 2001. On feature distributional clustering for text categorization. In Proceedings of the 24th annual international ACM SIGIR conference on Research and development in information retrieval. ACM, 146–153.

[13]	Daniel M Bikel, Scott Miller, Richard Schwartz, and Ralph Weischedel. 1997. Nymble: a high-performance learning name-finder. In Proceedings of the fifth conference on Applied natural language processing. Association for Computational Linguistics, 194–201.

[14]	David M Blei, Thomas L Griffiths, Michael I Jordan, and Joshua B Tenenbaum. 2003. Hierarchical Topic Models and the Nested Chinese Restaurant Process.. In NIPS, Vol. 16.

[15]	David M Blei and Jon D McAuliffe. 2007. Supervised Topic Models.. In NIPS, Vol. 7. 121–128.

[16]	David M Blei, Andrew Y Ng, and Michael I Jordan. 2003. Latent dirichlet allocation. the Journal of machine Learning research 3 (2003), 993–1022.

[17]	Olivier Bodenreider. 2004. The unified medical language system (UMLS): integrat-ing biomedical terminology. Nucleic acids research 32, suppl 1 (2004), D267–D270.

[18]	Paul S Bradley and Usama M Fayyad. 1998. Refining Initial Points for K-Means Clustering.. In ICML, Vol. 98. Citeseer, 91–99.

[19]	Leo Breiman, Jerome Friedman, Charles J Stone, and Richard A Olshen. 1984. Classification and regression trees. CRC press.

[20]	Markus Bundschus, Mathaeus Dejori, Martin Stetter, Volker Tresp, and Hans-Peter Kriegel. 2008. Extraction of semantic biomedical relations from text using conditional random fields. BMC bioinformatics 9, 1 (2008), 207.

[21]	Christopher JC Burges. 1998. A tutorial on support vector machines for pattern recognition. Data mining and knowledge discovery 2, 2 (1998), 121–167.

[22]Igor Cadez, David Heckerman, Christopher Meek, Padhraic Smyth, and Steven White. 2003. Model-based clustering and visualization of navigation patterns on a web site. Data Mining and Knowledge Discovery 7, 4 (2003), 399–424.

[23]Bob Carpenter. 2010. Integrating out multinomial parameters in latent Dirichlet al-location and naive bayes for collapsed Gibbs sampling. Technical Report. Technical report, LingPipe.

[24]	Soumen Chakrabarti, Byron Dom, Rakesh Agrawal, and Prabhakar Raghavan. 1997. Using taxonomy, discriminants, and signatures for navigating in text databases. In VLDB, Vol. 97. 446–455.

[25]	Yee Seng Chan and Dan Roth. 2010. Exploiting background knowledge for relation extraction. In Proceedings of the 23rd International Conference on Computational Linguistics. Association for Computational Linguistics, 152–160.

[26]	Yee Seng Chan and Dan Roth. 2011. Exploiting syntactico-semantic structures for relation extraction. In Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies-Volume 1. Association for Computational Linguistics, 551–560.

[27]	Chaitanya Chemudugunta, America Holloway, Padhraic Smyth, and Mark Steyvers. 2008. Modeling documents by combining semantic concepts with unsupervised statistical learning. In The Semantic Web-ISWC 2008. Springer, 229–244.

[28]	Ming-Syan Chen, Jiawei Han, and Philip S. Yu. 1996. Data mining: an overview from a database perspective. IEEE Transactions on Knowledge and data Engineering 8, 6 (1996), 866–883.

[29]	Hai Leong Chieu and Hwee Tou Ng. 2003. Named Entity Recognition with a Maximum Entropy Approach. In Proceedings of the Seventh Conference on Natural Language Learning at HLT-NAACL 2003 - Volume 4 (CONLL ’03). Association for Computational Linguistics, Stroudsburg, PA, USA, 160–163. https://doi.org/10. 3115/1119176.1119199

[30]	Aaron M Cohen and William R Hersh. 2005. A survey of current work in biomedical text mining. Briefings in bioinformatics 6, 1 (2005), 57–71.

[31]	K Bretonnel Cohen and Lawrence Hunter. 2008. Getting started in text mining. PLoS computational biology 4, 1 (2008), e20.

[32]	Nigel Collier, Chikashi Nobata, and Jun-ichi Tsujii. 2000. Extracting the names of genes and gene products with a hidden Markov model. In Proceedings of the 18th conference on Computational linguistics-Volume 1. Association for Computational Linguistics, 201–207.

[33]	Peter Corbett and Ann Copestake. 2008. Cascaded classifiers for confidence-based chemical named entity recognition. BMC bioinformatics 9, Suppl 11 (2008), S4.

[34]	Corinna Cortes and Vladimir Vapnik. 1995. Support-vector networks. Machine learning 20, 3 (1995), 273–297.

[35]	Jim Cowie and Wendy Lehnert. 1996. Information extraction. Commun. ACM 39, 1 (1996), 80–91.

[36]	Douglass R Cutting, David R Karger, and Jan O Pedersen. 1993. Constant interaction-time scatter/gather browsing of very large document collections. In Proceedings of the 16th annual international ACM SIGIR conference on Research and development in information retrieval. ACM, 126–134.

[37]	Douglass R Cutting, David R Karger, Jan O Pedersen, and John W Tukey. 1992. Scatter/gather: A cluster-based approach to browsing large document collections. In Proceedings of the 15th annual international ACM SIGIR conference on Research and development in information retrieval. ACM, 318–329.

[38]	Dipanjan Das and André FT Martins. 2007. A survey on automatic text sum-marization. Literature Survey for the Language and Statistics II course at CMU 4 (2007), 192–195.

[39]	Mahmood Doroodchi, Azadeh Iranmehr, and Seyed Amin Pouriyeh. 2009. An investigation on integrating XML-based security into Web services. In GCC Conference & Exhibition, 2009 5th IEEE. IEEE, 1–5.

[40]	Harris Drucker, S Wu, and Vladimir N Vapnik. 1999. Support vector machines for spam categorization. Neural Networks, IEEE Transactions on 10, 5 (1999), 1048–1054.

[41]	Richard O Duda, Peter E Hart, and David G Stork. 2012. Pattern classification. John Wiley & Sons.

[42]	S Dumais, G Furnas, T Landauer, S Deerwester, S Deerwester, et al. 1995. Latent semantic indexing. In Proceedings of the Text Retrieval Conference.

[43]	Sašo Džeroski. 2009. Relational data mining. In Data Mining and Knowledge Discovery Handbook. Springer, 887–911.

[44]	Christos Faloutsos and Douglas W Oard. 1998. A survey of information retrieval and filtering methods. Technical Report.

[45]	Usama M Fayyad, Gregory Piatetsky-Shapiro, Padhraic Smyth, et al. 1996. Knowl-edge Discovery and Data Mining: Towards a Unifying Framework.. In KDD, Vol. 96. 82–88.

[46]	Ronen Feldman and Ido Dagan. 1995. Knowledge Discovery in Textual Databases (KDT).. In KDD, Vol. 95. 112–117.

[47]	Guozhong Feng, Jianhua Guo, Bing-Yi Jing, and Lizhu Hao. 2012. A Bayesian feature selection paradigm for text classification. Information Processing & Man-agement 48, 2 (2012), 283–302.

[48]	William J Frawley, Gregory Piatetsky-Shapiro, and Christopher J Matheus. 1992. Knowledge discovery in databases: An overview. AI magazine 13, 3 (1992), 57.

[49]	Yoav Freund and Robert E Schapire. 1997. A decision-theoretic generalization of on-line learning and an application to boosting. Journal of computer and system sciences 55, 1 (1997), 119–139.

[50]	Mark A Friedl and Carla E Brodley. 1997. Decision tree classification of land cover from remotely sensed data. Remote sensing of environment 61, 3 (1997), 399–409.

[51]	Robert Gaizauskas, George Demetriou, Peter J. Artymiuk, and Peter Willett. 2003. Protein structures and information extraction from biological texts: the PASTA system. Bioinformatics 19, 1 (2003), 135–143.

[52]	John Gantz and David Reinsel. 2012. THE DIGITAL UNIVERSE IN 2020: Big Data, Bigger Digital Shadows, and Biggest Grow th in the Far East. Tech-nical Report 1. IDC, 5 Speen Street, Framingham, MA 01701 USA. Ac-cessed online on May, 2017. https://www.emc.com/collateral/analyst-reports/idc-the-digital-universe-in-2020.pdf.

[53]	Walter R Gilks, Sylvia Richardson, and David J Spiegelhalter. 1996. Introducing markov chain monte carlo. In Markov chain Monte Carlo in practice. Springer, 1–19.

[54]	Siddharth Gopal and Yiming Yang. 2010. Multilabel classification with meta-level features. In Proceedings of the 33rd international ACM SIGIR conference on Research and development in information retrieval. ACM, 315–322.

[55]	Thomas L Griffiths and Mark Steyvers. 2002. A probabilistic approach to semantic representation. In Proceedings of the 24th annual conference of the cognitive science society. Citeseer, 381–386.

[56]	Thomas L Griffiths and Mark Steyvers. 2004. Finding scientific topics. Proceedings of the National academy of Sciences of the United States of America 101, Suppl 1 (2004), 5228–5235.

[57]	Serkan Günal, Semih Ergin, M Bilginer Gülmezoğlu, and Ö Nezih Gerek. 2006. On feature extraction for spam e-mail detection. In Multimedia Content Repre-sentation, Classification and Security. Springer, 635–642.

[58]	Pritam Gundecha and Huan Liu. 2012. Mining social media: a brief introduction. In New Directions in Informatics, Optimization, Logistics, and Production. Informs, 1–17.

[59]	Zhou GuoDong, Su Jian, Zhang Jie, and Zhang Min. 2005. Exploring various knowledge in relation extraction. In Proceedings of the 43rd annual meeting on as-sociation for computational linguistics. Association for Computational Linguistics, 427–434.

[60]	Juan B Gutierrez, Mary R Galinski, Stephen Cantrell, and Eberhard O Voit. 2015. From within host dynamics to the epidemiology of infectious disease: scientific overview and challenges. (2015).

[61]	Eui-Hong Sam Han, George Karypis, and Vipin Kumar. 2001. Text categorization using weight adjusted k-nearest neighbor classification. Springer.
[62]	Jiawei Han, Micheline Kamber, and Jian Pei. 2006. Data mining: concepts and techniques. Morgan kaufmann.

[63]	Xianpei Han and Le Sun. 2012. An entity-topic model for entity linking. In Proceedings of the 2012 Joint Conference on Empirical Methods in Natural Lan-guage Processing and Computational Natural Language Learning. Association for Computational Linguistics, 105–115.

[64]	Gregor Heinrich. 2005. Parameter estimation for text analysis. Technical Report. Technical report.

[65]	William Hersh. 2008. Information retrieval: a health and biomedical perspective. Springer Science & Business Media.

[66]	Thomas Hofmann. 1999. Probabilistic latent semantic indexing. In Proceedings of the 22nd annual international ACM SIGIR conference on Research and development in information retrieval. ACM, 50–57.

[67]	Andreas Hotho, Andreas Nürnberger, and Gerhard Paaß. 2005. A Brief Survey of Text Mining.. In Ldv Forum, Vol. 20. 19–62.

[68]	David A Hull et al. 1996. Stemming algorithms: A case study for detailed evalua-tion. JASIS 47, 1 (1996), 70–84.

[69]	Hideki Isozaki and Hideto Kazawa. 2002. Efficient support vector classifiers for named entity recognition. In Proceedings of the 19th international conference on Computational linguistics-Volume 1. Association for Computational Linguistics, 1–7.

[70]	Anil K Jain and Richard C Dubes. 1988. Algorithms for clustering data. Prentice-Hall, Inc.

[71]	Mike James. 1985. Classification algorithms. Wiley-Interscience.

[72]	Jing Jiang and ChengXiang Zhai. 2007. A Systematic Exploration of the Feature Space for Relation Extraction.. In HLT-NAACL. 113–120.

[73]	Thorsten Joachims. 1996. A Probabilistic Analysis of the Rocchio Algorithm with TFIDF for Text Categorization. Technical Report. DTIC Document.

[74]	Thorsten Joachims. 1998. Text categorization with support vector machines: Learn-ing with many relevant features. Springer.

[75]	Thorsten Joachims. 2001. A statistical learning learning model of text classifica-tion for support vector machines. In Proceedings of the 24th annual international ACM SIGIR conference on Research and development in information retrieval. ACM, 128–136.

[76]	Tom Kalt and WB Croft. 1996. A new probabilistic model of text classification and retrieval. Technical Report. Citeseer.

[77]	Nanda Kambhatla. 2004. Combining lexical, syntactic, and semantic features with maximum entropy models for extracting relations. In Proceedings of the ACL 2004 on Interactive poster and demonstration sessions. Association for Computational Linguistics, 22.

[78]	Mehmed Kantardzic. 2011. Data mining: concepts, models, methods, and algorithms. John Wiley & Sons.

[79]	Tapas Kanungo, David M Mount, Nathan S Netanyahu, Christine D Piatko, Ruth Silverman, and Angela Y Wu. 2002. An efficient k-means clustering algorithm: Analysis and implementation. Pattern Analysis and Machine Intelligence, IEEE Transactions on 24, 7 (2002), 881–892.

[80]	Anne Kao and Stephen R Poteet. 2007. Natural language processing and text mining. Springer.

[81]	Saurabh S Kataria, Krishnan S Kumar, Rajeev R Rastogi, Prithviraj Sen, and Srinivasan H Sengamedu. 2011. Entity disambiguation with hierarchical topic models. In Proceedings of the 17th ACM SIGKDD international conference on Knowledge discovery and data mining. ACM, 1037–1045.

[82]	Leonard Kaufman and Peter J Rousseeuw. 2009. Finding groups in data: an introduction to cluster analysis. Vol. 344. John Wiley & Sons.

[83]	Jun’ichi Kazama, Takaki Makino, Yoshihiro Ohta, and Jun’ichi Tsujii. 2002. Tuning support vector machines for biomedical named entity recognition. In Proceedings of the ACL-02 workshop on Natural language processing in the biomedical domain-Volume 3. Association for Computational Linguistics, 1–8.

[84]	Daphne Koller and Mehran Sahami. 1997. Hierarchically classifying documents using very few words. (1997).

[85]	John Lafferty, Andrew McCallum, and Fernando CN Pereira. 2001. Conditional random fields: Probabilistic models for segmenting and labeling sequence data.(2001).

[86]	Leah S Larkey and W Bruce Croft. 1996. Combining classifiers in text catego-rization. In Proceedings of the 19th annual international ACM SIGIR conference on Research and development in information retrieval. ACM, 289–297.

[87]	Ulf Leser and Jörg Hakenberg. 2005. What makes a gene name? Named entity recognition in the biomedical literature. Briefings in Bioinformatics 6, 4 (2005).

[88]	David D Lewis. 1998. Naive (Bayes) at forty: The independence assumption in information retrieval. In Machine learning: ECML-98. Springer, 4–15.

[89]	Xin Li and Dan Roth. 2002. Learning question classifiers. In Proceedings of the 19th international conference on Computational linguistics-Volume 1. Association for Computational Linguistics, 1–7.

[90]	Elizabeth D Liddy. 2001. Natural language processing. (2001).

[91]	Anthony ML Liekens, Jeroen De Knijf, Walter Daelemans, Bart Goethals, Pe-ter De Rijk, and Jurgen Del-Favero. 2011. BioGraph: unsupervised biomedical knowledge discovery via automated hypothesis generation. Genome biology 12, 6 (2011), R57.

[92]	Julie B Lovins. 1968. Development of a stemming algorithm. MIT Information Processing Group, Electronic Systems Laboratory.

[93]	Christopher D Manning, Prabhakar Raghavan, and Hinrich Schütze. 2008. Intro-duction to information retrieval. Vol. 1. Cambridge university press Cambridge.

[94]	Christopher D Manning, Hinrich Schütze, et al. 1999. Foundations of statistical natural language processing. Vol. 999. MIT Press.

[95]	AT Mc Cray. 1993. A. The Unified Medical Language System. Meth Inf Med 34 (1993), 281–291.

[96]	Andrew McCallum, Kamal Nigam, et al. 1998. A comparison of event models for naive bayes text classification. In AAAI-98 workshop on learning for text categorization, Vol. 752. Citeseer, 41–48.

[97]	Andrew McCallum, Ronald Rosenfeld, Tom M Mitchell, and Andrew Y Ng. 1998. Improving Text Classification by Shrinkage in a Hierarchy of Classes.. In ICML, Vol. 98. 359–367.

[98]	Andrew Kachites McCallum. 1996. Bow: A toolkit for statistical language model-ing, text retrieval, classification and clustering. (1996).

[99]	Andrew Kachites McCallum. 2002. Mallet: A machine learning for language toolkit. (2002).

[100]	David Mimno, Wei Li, and Andrew McCallum. 2007. Mixtures of hierarchical topics with pachinko allocation. In Proceedings of the 24th international conference on Machine learning. ACM, 633–640.

[101]	Tom M Mitchell. 1997. Machine learning. 1997. Burr Ridge, IL: McGraw Hill 45 (1997).

[102]	Tomohiro Mitsumori, Sevrani Fation, Masaki Murata, Kouichi Doi, and Hirohumi Doi. 2005. Gene/protein name recognition based on support vector machine using dictionary as features. BMC bioinformatics 6, Suppl 1 (2005), S8.

[103]	Fionn Murtagh. 1983. A survey of recent advances in hierarchical clustering algorithms. Comput. J. 26, 4 (1983), 354–359.

[104]	Fionn Murtagh. 1984. Complexities of hierarchic clustering algorithms: State of the art. Computational Statistics Quarterly 1, 2 (1984), 101–113.

[105]	Amir Netz, Surajit Chaudhuri, Jeff Bernhardt, and Usama Fayyad. 2000. In-tegration of data mining and relational databases. In Proceedings of the 26th International Conference on Very Large Databases, Cairo, Egypt. 285–296.

[106]	Kamal Nigam, Andrew McCallum, Sebastian Thrun, and Tom Mitchell. 1998. Learning to classify text from labeled and unlabeled documents. AAAI/IAAI 792 (1998).

[107]	Chikashi Nobata, Nigel Collier, and Jun-ichi Tsujii. 1999. Automatic term identification and classification in biology texts. In Proc. of the 5th NLPRS. Citeseer, 369–374.

[108]	Edgar Osuna, Robert Freund, and Federico Girosi. 1997. Training support vector machines: an application to face detection. In Computer Vision and Pattern Recognition, 1997. Proceedings., 1997 IEEE Computer Society Conference on. IEEE, 130–136.

[109]	Bo Pang and Lillian Lee. 2008. Opinion mining and sentiment analysis. Founda-tions and trends in information retrieval 2, 1-2 (2008), 1–135.

[110]	Martin F Porter. 1980. An algorithm for suffix stripping. Program: electronic library and information systems 14, 3 (1980), 130–137.

[111]	Seyed Amin Pouriyeh and Mahmood Doroodchi. 2009. Secure SMS Banking Based On Web Services. In SWWS. 79–83.

[112]	Seyed Amin Pouriyeh, Mahmood Doroodchi, and MR Rezaeinejad. 2010. Secure Mobile Approaches Using Web Services.. In SWWS. 75–78.

[113]	J. Ross Quinlan. 1986. Induction of decision trees. Machine learning 1, 1 (1986), 81–106.

[114]	Lawrence Rabiner. 1989. A tutorial on hidden Markov models and selected applications in speech recognition. Proc. IEEE 77, 2 (1989), 257–286.

[115]	Dragomir R Radev, Eduard Hovy, and Kathleen McKeown. 2002. Introduction to the special issue on summarization. Computational linguistics 28, 4 (2002), 399–408.

[116]	Martin Rajman and Romaric Besançon. 1998. Text mining: natural language techniques and text mining applications. In Data Mining and Reverse Engineering. Springer, 50–64.

[117]	Payam Porkar Rezaeiye, Mojtaba Sedigh Fazli, et al. 2014. Use HMM and KNN for classifying corneal data. arXiv preprint arXiv:1401.7486 (2014).

[118]	Mehran Sahami, Susan Dumais, David Heckerman, and Eric Horvitz. 1998. A Bayesian approach to filtering junk e-mail. In Learning for Text Categorization: Papers from the 1998 workshop, Vol. 62. 98–105.

[119]	Hassan Saif, Miriam Fernández, Yulan He, and Harith Alani. 2014. On stopwords, filtering and data sparsity for sentiment analysis of twitter. (2014).

[120]	Gerard Salton and Christopher Buckley. 1988. Term-weighting approaches in automatic text retrieval. Information processing & management 24, 5 (1988), 513–523.

[121]	Gerard Salton, Anita Wong, and Chung-Shu Yang. 1975. A vector space model for automatic indexing. Commun. ACM 18, 11 (1975), 613–620.

[122]	Sunita Sarawagi et al. 2008. Information extraction. Foundations and Trends® in Databases 1, 3 (2008), 261–377.

[123]	Yutaka Sasaki, Yoshimasa Tsuruoka, John McNaught, and Sophia Ananiadou. 2008. How to make the most of NE dictionaries in statistical NER. BMC bioinfor-matics 9, Suppl 11 (2008), S5.

[124]	Guergana K Savova, James J Masanz, Philip V Ogren, Jiaping Zheng, Sunghwan Sohn, Karin C Kipper-Schuler, and Christopher G Chute. 2010. Mayo clinical Text Analysis and Knowledge Extraction System (cTAKES): architecture, compo-nent evaluation and applications. Journal of the American Medical Informatics Association 17, 5 (2010), 507–513.

[125]	Robert E Schapire and Yoram Singer. 2000. BoosTexter: A boosting-based system for text categorization. Machine learning 39, 2-3 (2000), 135–168.

[126]	Fabrizio Sebastiani. 2002. Machine learning in automated text categorization. ACM computing surveys (CSUR) 34, 1 (2002), 1–47.

[127]	Prithviraj Sen. 2012. Collective context-aware topic models for entity disam-biguation. In Proceedings of the 21st international conference on World Wide Web. ACM, 729–738.

[128]	Burr Settles. 2004. Biomedical named entity recognition using conditional random fields and rich feature sets. In Proceedings of the International Joint Workshop on Natural Language Processing in Biomedicine and its Applications. Association for Computational Linguistics, 104–107.

[129]	Dan Shen, Jie Zhang, Guodong Zhou, Jian Su, and Chew-Lim Tan. 2003. Effec-tive adaptation of a hidden markov model-based named entity recognizer for biomedical domain. In Proceedings of the ACL 2003 workshop on Natural language processing in biomedicine-Volume 13. Association for Computational Linguistics, 49–56.

[130]	Catarina Silva and Bernardete Ribeiro. 2003. The importance of stop word re-moval on recall values in text categorization. In Neural Networks, 2003. Proceedings of the International Joint Conference on, Vol. 3. IEEE, 1661–1666.

[131]	Amit Singhal. 2001. Modern information retrieval: A brief overview. IEEE Data Eng. Bull. 24, 4 (2001), 35–43.

[132]	Michael Steinbach, George Karypis, Vipin Kumar, et al. 2000. A comparison of document clustering techniques. In KDD workshop on text mining, Vol. 400. Boston, 525–526.

[133]	Mark Steyvers and Tom Griffiths. 2007. Probabilistic topic models. Handbook of latent semantic analysis 427, 7 (2007), 424–440.

[134]	Koichi Takeuchi and Nigel Collier. 2005. Bio-medical entity extraction using support vector machines. Artificial Intelligence in Medicine 33, 2 (2005), 125–137.

[135]	Songbo Tan, Yuefen Wang, and Gaowei Wu. 2011. Adapting centroid classifier for document categorization. Expert Systems with Applications 38, 8 (2011), 10264–10273.

[136]	E. D. Trippe, J. B. Aguilar, Y. H. Yan, M. V. Nural, J. A. Brady, M. Assefi, S. Safaei,M.	Allahyari, S. Pouriyeh, M. R. Galinski, J. C. Kissinger, and J. B. Gutierrez. 2017. A Vision for Health Informatics: Introducing the SKED Framework An Extensible Architecture for Scientific Knowledge Extraction from Data. ArXiv e-prints (2017). arXiv:1706.07992

[137]	Stéphane Tufféry. 2011. Data mining and statistics for decision making. John Wiley & Sons.

[138]	Howard Turtle and W Bruce Croft. 1989. Inference networks for document retrieval. In Proceedings of the 13th annual international ACM SIGIR conference on Research and development in information retrieval. ACM, 1–24.

[139]	Yu Usami, Han-Cheol Cho, Naoaki Okazaki, and Jun’ichi Tsujii. 2011. Automatic acquisition of huge training data for bio-medical named entity recognition. In Proceedings of BioNLP 2011 Workshop. Association for Computational Linguistics, 65–73.

[140]	Alper Kursat Uysal and Serkan Gunal. 2014. The impact of preprocessing on text classification. Information Processing & Management 50, 1 (2014), 104–112.

[141]	Vladimir Vapnik. 1982. Estimation of Dependences Based on Empirical Data: Springer Series in Statistics (Springer Series in Statistics). Springer-Verlag New York, Inc., Secaucus, NJ, USA.

[142]	Vladimir Vapnik. 2000. The nature of statistical learning theory. springer.

[143]	Jonathan J Webster and Chunyu Kit. 1992. Tokenization as the initial phase in NLP. In Proceedings of the 14th conference on Computational linguistics-Volume 4. Association for Computational Linguistics, 1106–1110.

[144]	Peter Willett. 1988. Recent trends in hierarchic document clustering: a critical review. Information Processing & Management 24, 5 (1988), 577–597.

[145]	Rui Xu, Donald Wunsch, et al. 2005. Survey of clustering algorithms. Neural Networks, IEEE Transactions on 16, 3 (2005), 645–678.

[146]	Christopher C Yang, Haodong Yang, Ling Jiang, and Mi Zhang. 2012. Social me-dia mining for drug safety signal detection. In Proceedings of the 2012 international workshop on Smart health and wellbeing. ACM, 33–40.

[147]	Yiming Yang and Xin Liu. 1999. A re-examination of text categorization methods. In Proceedings of the 22nd annual international ACM SIGIR conference on Research and development in information retrieval. ACM, 42–49.

[148]	Illhoi Yoo and Min Song. 2008. Biomedical Ontologies and Text Mining for Biomedicine and Healthcare: A Survey. JCSE 2, 2 (2008), 109–136.

[149]	Liyang Yu. 2011. A developer’s guide to the semantic Web. Springer.

[150]	Shaojun Zhao. 2004. Named entity recognition in biomedical texts using an HMM model. In Proceedings of the international joint workshop on natural language processing in biomedicine and its applications. Association for Computational Linguistics, 84–87.

[151]	Ying Zhao and George Karypis. 2004. Empirical and theoretical comparisons of selected criterion functions for document clustering. Machine Learning 55, 3 (2004), 311–331.