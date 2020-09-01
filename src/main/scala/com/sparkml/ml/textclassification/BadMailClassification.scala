package com.sparkml.ml.textclassification

/**
 * @Author:
 * @Date: 2020-07-24 11:18
 * @Version: 1.0
 * @Modified By:
 * @Description:
 */
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, Word2Vec}
import org.apache.spark.sql.SparkSession

/*
从 HDFS 上读取原始数据集，并创建一个 DataFrame。
使用 StringIndexer 将原始的文本标签 (“Ham”或者“Spam”) 转化成数值型的表型，以便 Spark ML 处理。
使用 Word2Vec 将短信文本转化成数值型词向量。
使用 MultilayerPerceptronClassifier 训练一个多层感知器模型。
使用 LabelConverter 将预测结果的数值标签转化成原始的文本标签。
最后在测试数据集上测试模型的预测精确度。

 */

// 参考链接：https://blog.csdn.net/weixin_32265569/article/details/107604269

object BadMailClassification {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.WARN)
    val spark = SparkSession.builder().appName("BadMailClassification").master("local").getOrCreate()

    //2， 创建 集并分词
    val parsedRDD = spark.sparkContext.textFile("data/ml/crossvalidation/垃圾邮件分类_SMSSpamCollection.txt")
      .map(_.split(" "))
      .filter( arrayRow => arrayRow.length > 2)
      .map(eachRow => {
      (eachRow(0), eachRow(1).split(" "))
    })
    val msgDF = spark.createDataFrame(parsedRDD).toDF("label", "message")

    //3， 将标签转化为索引值
    val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(msgDF)
    //4， 创建Word2Vec，分词向量大小100
    val VECTOR_SIZE = 100

    val word2Vec = new Word2Vec()
      .setInputCol("message")
      .setOutputCol("features")
      .setVectorSize(VECTOR_SIZE)
      .setMinCount(1)

    //5， 创建多层感知器
    //输入层VECTOR_SIZE个，中间层两层分别是6，,5个神经元，输出层2个
    val layers = Array[Int](VECTOR_SIZE, 6, 5, 2)
    val mlpc = new MultilayerPerceptronClassifier()
      /*
          layers：这个参数是一个整型数组类型，
          第一个元素需要和特征向量的维度相等，
          最后一个元素需要训练数据的标签取值个数相等，如 2 分类问题就写 2。
          中间的元素有多少个就代表神经网络有多少个隐层，元素的取值代表了该层的神经元的个数。例如val layers = Array[Int](100,6,5,2)。
       */
      .setLayers(layers)
      .setBlockSize(512) // 该参数被前馈网络训练器用来将训练样本数据的每个分区都按照 blockSize 大小分成不同组，并且每个组内的每个样本都会被叠加成一个向量，以便于在各种优化算法间传递。该参数的推荐值是 10-1000，默认值是 128。
      .setSeed(1234L)
      .setMaxIter(128) // 优化算法求解的最大迭代次数。默认值是 100。
      .setFeaturesCol("features") // 输入数据 DataFrame 中指标特征列的名称。
      .setLabelCol("indexedLabel") // 输入数据 DataFrame 中标签列的名称。
      .setPredictionCol("prediction") // 预测结果的列名称。
      .setTol(10) // 优化算法迭代求解过程的收敛阀值。默认值是 1e-4。不能为负数。
      .setStepSize(0.025) // 优化算法的每一次迭代的学习速率。默认值是0.025


    //6， 将索引转换为原有标签
    val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)

    //7， 数据集分割
    val Array(trainingData, testData) = msgDF.randomSplit(Array(0.8, 0.2))

    //8， 创建pipeline并训练数据
    val pipeline = new Pipeline().setStages(Array(labelIndexer, word2Vec, mlpc, labelConverter))
    val model = pipeline.fit(trainingData)
    val predictionResultDF = model.transform(testData)

    //below 2 lines are for debug use
    predictionResultDF.printSchema
    predictionResultDF.select("message", "label", "predictedLabel").show(30)

    //9， 评估训练结果
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("indexedLabel")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")

    val predictionAccuracy = evaluator.evaluate(predictionResultDF)
    println("Testing Accuracy is %2.4f".format(predictionAccuracy * 100) + "%")


    //将测试集数据交给管道进行测试
//    val predictions = model.transform(testData)
//    predictions.select("message","label", "predictedLabel").show(false)
//    val evaluator = new MulticlassClassificationEvaluator()
//      .setLabelCol("label")
//      .setPredictionCol("predictedLabel")
//      .setMetricName("accuracy")
//    val accuracy = evaluator.evaluate(predictions)
//    println(s"正确率 = ${accuracy}" + s"\t错误率 = ${(1.0 - accuracy)}")
  }
}
