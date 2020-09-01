package com.sparkml.ml.textclassification

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.SparkSession

/**
 *
 */
object TextClassificationParam {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.WARN)
    val spark = SparkSession.builder().appName("TF_IDF_Demo").master("local").getOrCreate()
    val df = spark.createDataFrame(Seq(
      ("Hadoop is a big data framework", 1), //Hadoop,is,a
      ("Spark is a big data framework", 0),
      ("Spark is a big data framework", 0)
    )).toDF("sentence", "label")

    //todo 定义一个统一的paramMap,用来封装管道中转换器和模型学习器相关参数
    val paramMap = ParamMap()

    //分词转换器   在spark ml中对应的转换器帮助我们进行转换,Tokenizer转换器帮助我们实现分词
    val tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words")

    //TF转换器  需要对wordsDF进行转换，就是进行TF词频统计
    val hashingTF = new HashingTF().setInputCol(tokenizer.getOutputCol).setOutputCol("tf") //.setNumFeatures(262144)
    paramMap.put(hashingTF.numFeatures -> 100)

    //IDF模型学习器（算法）  这个IDF不是转换器，而是模型学习器（算法）
    val idf = new IDF().setInputCol(hashingTF.getOutputCol).setOutputCol("features")

    // 模型学习器(就是逻辑回归算法)
    val lr = new LogisticRegression()
    //设置最大迭代次数
    //.setMaxIter(20)
    //正则化参数
    //.setRegParam(0.01)
    paramMap.put(lr.maxIter -> 30).put(lr.regParam -> 0.02)

    //pipeline也是一个模型学习器,可以设置多个stages，每个stage要么是转换器，要么是模型学习器
    val pipeline = new Pipeline().setStages(Array(tokenizer, hashingTF, idf, lr))

    val pipelineModel = pipeline.fit(df, paramMap) //模型学习器调用fit方法训练完成后，返回的是一个转换器

    val testDF = spark.createDataFrame(Seq(
      ("1111", "Hadoop is a big data ecosystem Technology")
    )).toDF("id", "sentence")

    //将测试集数据交给管道进行测试
    pipelineModel.transform(testDF).show(false)

    spark.stop()
  }
}
