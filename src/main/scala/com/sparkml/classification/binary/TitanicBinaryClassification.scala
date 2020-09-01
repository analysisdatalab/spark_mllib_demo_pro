package com.sparkml.classification.binary

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.classification._
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

/**
 *
 * 分类算法--二分类算法--Titanic号案例--乘客生成状况预测分析
 *
 * 1，加载数据
 * 2，提取特征工程
 * 3，将提取的特征工程数据交给算法得到模型
 * 4，使用模型进行预测
 *
 */
object TitanicBinaryClassification {
  def main(args: Array[String]): Unit = {
    //设置日志级别
    Logger.getLogger("org").setLevel(Level.WARN)
    val spark = SparkSession.builder().appName("TitanicBinaryClassification").master("local").getOrCreate()

    //todo 1,加载数据
    val df = spark.read
      //第一行作为列的名称
      .option("header", "true")
      //自动推断出每一列的数据类型
      //.option("inferSchema", "true")
      .csv("data/classification/train.csv")


    //df.printSchema()
    //df.show(10, true)

    //提取出需要的字段(乘客的特征)"Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"
    val df2 = df.select("Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare")
      //.show(10, true)

    //计算乘客的平均年龄
    val avgAge = df2.select("Age").agg("Age" -> "avg").first().getDouble(0)

    //todo 2，提取特征工程
    /**
     * 提取的特征数据需要封装到 LabeledPoint 这个类中
     * case class LabeledPoint @Since("1.0.0") (
     * label: Double,  标签
     * features: Vector 特征，在当前这个案例中指代的是乘客的特征。需要注意的是features是一个向量
     * )
     *
     * Vector：是一个向量，我们可以通过Vectors提供的方法来创建稀疏向量或稠密向量
     */
    val labeledPointRDD = df2.rdd.map(row => {
      //标签（乘客生存状况）
      val label = row.getAs[String]("Survived").toDouble
      //客舱等级
      val pclass = row.getAs[String]("Pclass").toDouble
      val sex = if ("male".equals(row.getAs[String]("Sex"))) 1.0 else 0.0
      val age = if (row.get(3) == null) avgAge else row.getAs[String]("Age").toDouble
      val sibsp = row.getAs[String]("SibSp").toDouble
      val parch = row.getAs[String]("Parch").toDouble
      val fare = row.getAs[String]("Fare").toDouble
      // 使用Vectors.dense创建稠密向量
      LabeledPoint(label, Vectors.dense(pclass, sex, age, sibsp, parch, fare))
    })

    /**
     * (1.0,[3.0,0.0,29.69911764705882,1.0,0.0,24.15])
     * (0.0,[1.0,1.0,47.0,0.0,0.0,52.0])
     * (0.0,[3.0,0.0,14.5,1.0,0.0,14.4542])
     */
    //labeledPointRDD.foreach(println)

    //将rdd进行持久化
    labeledPointRDD.cache()

    //将乘客特征数据划分成训练集和测试集
    val Array(trainRDD, testRDD) = labeledPointRDD.randomSplit(Array(0.8, 0.2))

    /**
     * todo 3，将提取的特征工程数据交个算法得到模型
     * a,支持向量机（linear SVMs）
     * b,逻辑回归算法（logistic regression）
     * c,决策树（decision trees）
     * d,朴素贝叶斯算法（naive Bayes）
     */

    //todo 使用支持向量机对乘客的生存状况进行预测
    val svmModel: SVMModel = SVMWithSGD.train(trainRDD, 100)
    val svmPredictAndActualRDD: RDD[(Double, Double)] = testRDD.map { case LabeledPoint(label, features) => (svmModel.predict(features), label) }
    // svmPredictAndActualRDD.foreach(println) // 前面是预测值，后面是实际值
    //如何评估模型的好坏，那么就需要一个模型评估器，来评判这个模型
    //可以使用二分类评估器对模型预测的结果进行评估
    val svmMetrics = new BinaryClassificationMetrics(svmPredictAndActualRDD)
    //可以使用roc面积这个指标来评判模型的预测效果，0.5< roc < 1; 如果roc面积接近0.5预测效果好，如果接近1预测效果不好
    val svmROC = svmMetrics.areaUnderROC()
    println("支持向量机 ROC = " + svmROC)


    //todo 使用逻辑回归算法（logistic regression）对乘客的生存状况进行预测
    val lr = new LogisticRegressionWithLBFGS()
      //设置样本中类别的个数，我们这里只有生、死两种类型
      .setNumClasses(2)
    //使用run方法进行训练，得到一个模型，将训练数据传入训练算法
    val lrModel: LogisticRegressionModel = lr.run(trainRDD)
    //使用模型进行预测
    val lrPredictAndActualRDD: RDD[(Double, Double)] = testRDD.map { case LabeledPoint(label, features) => (lrModel.predict(features), label) }
    //使用二分类评估器对模型预测的结果进行评估
    val lrMetrics = new BinaryClassificationMetrics(lrPredictAndActualRDD)
    //可以使用roc面积这个指标来评判模型的预测效果，0.5< roc < 1
    val lrROC = lrMetrics.areaUnderROC()
    println("逻辑回归 ROC = " + lrROC)


    //todo 使用决策树（decision trees）对乘客的生存状况进行预测
    /**
     * def trainClassifier(
     * input: RDD[LabeledPoint],//训练集数据
     * numClasses: Int,//类别个数
     * categoricalFeaturesInfo: Map[Int, Int],//样本特征数据中，是否包含特征数据，如果包含需要告知
     * impurity: String, //不纯度度量方式，选择特征作为决策的依据。度量方式有两种：1，基尼系数 gini  2，香浓熵 entropy
     * maxDepth: Int,//数据的深度
     * maxBins: Int//树的叶子节点数 ，一般是2的n次方，如果是2，那么这颗决策树是一个二叉树
     * ): DecisionTreeModel
     */
    // 如果使用决策树也可以做回归问题预测，则调用trainRegressor
    val dtModel: DecisionTreeModel = DecisionTree.trainClassifier(trainRDD, 2, Map[Int, Int](), "gini", 6, 2)
    //使用模型进行预测
    val dtPredictAndActualRDD: RDD[(Double, Double)] = testRDD.map { case LabeledPoint(label, features) => (dtModel.predict(features), label) }
    //使用二分类评估器对模型预测的结果进行评估
    val dtMetrics = new BinaryClassificationMetrics(dtPredictAndActualRDD)
    //可以使用roc面积这个指标来评判模型的预测效果，0.5< roc < 1
    val dtROC = dtMetrics.areaUnderROC()
    println("决策树 ROC = " + dtROC)

    //todo 使用朴素贝叶斯算法（naive Bayes）对乘客生存状况进行预测
    //贝叶斯算法一般用于 文本分类/垃圾文本过滤/情感分类
    //第二个参数是正则化参数: 0.01~1
    val nbModel = NaiveBayes.train(trainRDD, 1.0)
    //使用模型进行预测
    val nbPredictAndActualRDD: RDD[(Double, Double)] = testRDD.map { case LabeledPoint(label, features) => (nbModel.predict(features), label) }
    //使用二分类评估器对模型预测的结果进行评估
    val nbMetrics = new BinaryClassificationMetrics(nbPredictAndActualRDD)
    //可以使用roc面积这个指标来评判模型的预测效果，0.5< roc < 1
    val nbROC = nbMetrics.areaUnderROC()
    println("素贝叶斯 ROC = " + nbROC)

    spark.stop()
  }
}
