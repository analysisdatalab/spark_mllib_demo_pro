package com.sparkml.classification.mult

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS, NaiveBayes}
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.configuration.{Algo, BoostingStrategy, Strategy}
import org.apache.spark.mllib.tree.impurity.Gini
import org.apache.spark.mllib.tree.loss.SquaredError
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.mllib.tree.{DecisionTree, GradientBoostedTrees, RandomForest}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.util

/**
 * 分类算法--多分类算法--鸢尾花案例
 */
object IrisMulticlassClassification {
  def main(args: Array[String]): Unit = {
    // todo 1.加载数据
    Logger.getLogger("org").setLevel(Level.WARN)
    val spark = SparkSession.builder().appName("IrisMulticlassClassification").master("local").getOrCreate()
    //加载鸢尾花数据
    val df = spark.read.option("inferSchema", "true").csv("data/classification/iris.data")
    //df.printSchema()
    df.show(10)  // 测试加载数据是否正常

    // todo 2.提取特征工程
    //LabeledPoint（类别，特征向量）
    //提取出鸢尾花的类别，然后给每个类别加上index角标，以index角标作为label
    val labelMap = df.rdd.map(row => row.getString(4)).distinct().zipWithIndex().collectAsMap()
    //println(labelMap)

    val labelMapBroadCast = spark.sparkContext.broadcast[collection.Map[String, Long]](labelMap)
    //Map(Iris-virginica -> 2, Iris-setosa -> 0, Iris-versicolor -> 1)
    //提取鸢尾花特征
    val labeledPointRDD = df.rdd.map(row => {
      //5.8,4.0,1.2,0.2,Iris-setosa
      val heX = row.getDouble(0)
      val heY = row.getDouble(1)
      val hbX = row.getDouble(2)
      val hbY = row.getDouble(3)
      val label = labelMapBroadCast.value(row.getString(4)).toDouble
      LabeledPoint(label, Vectors.dense(Array(heX, heY, hbX, hbY)))
    })

    labeledPointRDD.cache()

    //将数据划分成训练集和测试集
    val Array(trainRDD, testRDD) = labeledPointRDD.randomSplit(Array(0.8, 0.2))
    //trainRDD.foreach(println)

    /**
     * 使用以下多分类算法，实现对鸢尾花类别的预测
     *
     * a,逻辑回归算法（logistic regression）
     * b,决策树（decision trees）
     * c,随机森林 (RandomForest)
     * d,朴素贝叶斯算法（naive Bayes）
     */
    //todo 使用逻辑回归进行预测
    val lrModel = new LogisticRegressionWithLBFGS().setNumClasses(labelMap.size).run(trainRDD)
    val lrPredictAndActualRDD = testRDD.map { case LabeledPoint(label, features) => (lrModel.predict(features), label) }
    //需要多分类评估器，对模型进行评估
    val lrMetrics = new MulticlassMetrics(lrPredictAndActualRDD)
    //对多分类模型进行评估，可以使用准确率，精准率，召回率
    //准确率 预测正取的总数/需要预测的总数
    println("逻辑回归 准确率accuracy = " + lrMetrics.accuracy + "\t精确率weightedPrecision:"
      + lrMetrics.weightedPrecision + "\t召回率weightedRecall:" + lrMetrics.weightedRecall)


    //todo 使用决策树进行预测
    val dtModel = DecisionTree.trainClassifier(trainRDD, labelMap.size, Map[Int, Int](), "gini", 8, 2)
    val dtPredictAndActualRDD = testRDD.map { case LabeledPoint(label, features) => (dtModel.predict(features), label) }
    //需要多分类评估器，对模型进行评估
    val dtMetrics = new MulticlassMetrics(dtPredictAndActualRDD)
    //对多分类模型进行评估，可以使用准确率，精准率，召回率
    //准确率 预测正取的总数/需要预测的总数
    println("决策树 准确率accuracy = " + dtMetrics.accuracy + "\t精确率weightedPrecision:"
      + dtMetrics.weightedPrecision + "\t召回率weightedRecall:" + dtMetrics.weightedRecall)

    //todo 使用随机森林进行预测
    val rfModel: RandomForestModel = RandomForest.trainClassifier(
      trainRDD,
      labelMap.size,
      Map[Int, Int](),
      5,
      "auto", "gini", 5, 2)
    //使用模型进行预测
    val rfPredictAndActualRDD: RDD[(Double, Double)] = testRDD.map { case LabeledPoint(label, features) => (rfModel.predict(features), label) }
    //需要多分类评估器，对模型进行评估
    val rfMetrics = new MulticlassMetrics(rfPredictAndActualRDD)
    //对多分类模型进行评估，可以使用准确率，精准率，召回率
    //准确率 预测正取的总数/需要预测的总数
    println("随机森林 准确率accuracy = " + rfMetrics.accuracy + "\t精确率weightedPrecision:"
      + rfMetrics.weightedPrecision + "\t召回率weightedRecall:" + rfMetrics.weightedRecall)

    //todo 贝叶斯算法进行预测
    val nbModel = NaiveBayes.train(trainRDD)
    //使用模型进行预测
    val nbPredictAndActualRDD: RDD[(Double, Double)] = testRDD.map { case LabeledPoint(label, features) => (nbModel.predict(features), label) }
    //需要多分类评估器，对模型进行评估
    val nbMetrics = new MulticlassMetrics(nbPredictAndActualRDD)
    //对多分类模型进行评估，可以使用准确率，精准率，召回率
    //准确率 预测正取的总数/需要预测的总数
    println("贝叶斯 准确率accuracy = " + nbMetrics.accuracy + "\t精确率weightedPrecision:"
      + nbMetrics.weightedPrecision + "\t召回率weightedRecall:" + nbMetrics.weightedRecall)


    spark.stop()

  }
}
