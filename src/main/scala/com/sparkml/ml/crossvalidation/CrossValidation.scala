package com.sparkml.ml.crossvalidation

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.{Model, Pipeline, PipelineModel}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{HashingTF, Tokenizer}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.SparkSession

/**
 *
 */
object CrossValidation {
  def main(args: Array[String]): Unit = {

    Logger.getLogger("org").setLevel(Level.WARN)
    val spark = SparkSession.builder().appName("CrossValidation").master("local").getOrCreate()

    val training = spark.createDataFrame(Seq(
      (0L, "a b c d e spark", 1.0),
      (1L, "b d", 0.0),
      (2L, "spark f g h", 1.0),
      (3L, "hadoop mapreduce", 0.0)
    )).toDF("id", "text", "label")


    //分词转换器
    val tokenizer = new Tokenizer()
      .setInputCol("text")
      .setOutputCol("words")

    //词频统计转换器，将单词出现的次数作为特征向量
    val hashingTF = new HashingTF()
      .setNumFeatures(1000)
      .setInputCol(tokenizer.getOutputCol)
      .setOutputCol("features")

    //模型训练器（模型学习器，算法）
    val lr = new LogisticRegression()


    //这是一个管道，管道其实就是一个模型学习器，这个管道有3个stage
    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, hashingTF, lr))

    //todo 为了得到最佳模型，我们需要进行交叉验证，参数的交叉验证，和数据集的交叉验证（）
    val paramMaps = new ParamGridBuilder()
      .addGrid(hashingTF.numFeatures, Array(100, 200))
      .addGrid(lr.maxIter, Array(10, 20))
      .addGrid(lr.regParam, Array(0.0, 0.01))
      .build()

    //创建模型评估器
    val binaryClassificationEvaluator = new BinaryClassificationEvaluator().setLabelCol("label").setRawPredictionCol("prediction")

    //模型学习器
    val crossValidator = new CrossValidator()
      //设置模型学习器，交叉验证的对象
      .setEstimator(pipeline)
      //参数交叉验证
      .setEstimatorParamMaps(paramMaps)
      //数据集交叉验证，数据集划分几个等分
      .setNumFolds(2)
      //设置模型评估器
      .setEvaluator(binaryClassificationEvaluator)

    //交叉验证模型，可以其进行预测我们的样本数据
    //模型学习器调用fit方法后会产生一个转换器，这个crossValidatorModel就是一个转换器
    val crossValidatorModel = crossValidator.fit(training)
    //todo 获取最佳模型
    val bestModel = crossValidatorModel.bestModel.asInstanceOf[PipelineModel]
    //获取管道中hashingTF转换器最佳参数
    println("hashingTF转换器最佳参数 " + bestModel.stages(1).extractParamMap())

    //获取管道中逻辑回归LogisticRegression模型学习器最佳参数
    println("LogisticRegression模型学习器最佳参数 " + bestModel.stages(2).extractParamMap())

    val test = spark.createDataFrame(Seq(
      (4L, "spark i j k", 1.0),
      (5L, "l m n", 0.0),
      (6L, "spark hadoop spark", 1.0),
      (7L, "apache hadoop", 0.0)
    )).toDF("id", "text", "label")

    //使用交叉验证模型，对需要预测的数据进行分类
    val resultDF = bestModel.transform(test)

    resultDF.show(false)

    val areaUnderROC = binaryClassificationEvaluator
      //设置对模型评估采用的评估指标roc
      .setMetricName("areaUnderROC").evaluate(resultDF.select("label", "prediction"))
    //0.5< roc  <  1.0
    println("areaUnderROC :  " + areaUnderROC)

    spark.stop()
  }
}
