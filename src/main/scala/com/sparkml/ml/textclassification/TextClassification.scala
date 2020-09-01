package com.sparkml.ml.textclassification

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
import org.apache.spark.sql.SparkSession

/**
 *
 */
object TextClassification {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.WARN)
    val spark = SparkSession.builder().appName("TF_IDF_Demo").master("local").getOrCreate()
    val df = spark.createDataFrame(Seq(
      ("Hadoop is a big data framework", 1), //Hadoop,is,a
      ("Spark is a big data framework", 0),
      ("Spark is a big data framework", 0)
    )).toDF("sentence", "label")

    //分词转换器   在spark ml中对应的转换器帮助我们进行转换,Tokenizer转换器帮助我们实现分词
    val tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words")

    //TF转换器  需要对wordsDF进行转换，就是进行TF词频统计
    val hashingTF = new HashingTF().setInputCol(tokenizer.getOutputCol).setOutputCol("tf")

    //IDF模型学习器（算法）  这个IDF不是转换器，而是模型学习器（算法）
    val idf = new IDF().setInputCol(hashingTF.getOutputCol).setOutputCol("features")

    // 模型学习器(就是逻辑回归算法)，逻辑回归算法输入列必须叫"features"
    val lr = new LogisticRegression()
      //设置最大迭代次数
      .setMaxIter(20)
      //正则化参数，防止过度拟合
      .setRegParam(0.01)

    //pipeline也是一个模型学习器,可以设置多个stages，每个stage要么是转换器，要么是模型学习器
    val pipeline = new Pipeline().setStages(Array(tokenizer, hashingTF, idf, lr))

    //对管道Pipeline进行持久化
    pipeline.write.overwrite().save("data/pipeplne")
    val pipeline1 = Pipeline.load("data/pipeplne")

    val pipelineModel = pipeline1.fit(df) //模型学习器调用fit方法训练完成后，返回的是一个转换器
    //也可以对模型进行持久化
    pipelineModel.write.overwrite().save("data/pipelinemodel")
    val pipelineModel1 = PipelineModel.load("data/pipelinemodel")


    val testDF = spark.createDataFrame(Seq(
      ("1111", "Hadoop is a big data ecosystem Technology")
    )).toDF("id", "sentence")

    //将测试集数据交给管道进行测试
    //pipelineModel.transform(testDF).show(false)

    //使用持久化后的PipelineModel对原始数据进行预测
    pipelineModel1.transform(testDF).select("sentence", "features").show(false)

    spark.stop()
  }
}
