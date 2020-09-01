package com.sparkml.ml.tf_idf

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
import org.apache.spark.sql.SparkSession

/**
 *
 */
object TF_IDF_Demo {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.WARN)
    val spark = SparkSession.builder().appName("TF_IDF_Demo").master("local").getOrCreate()
    val df = spark.createDataFrame(Seq(
      ("Hadoop is a big data framework", 1), //Hadoop,is,a
      ("Spark is a big data framework", 0),
      ("Spark is a big data framework", 0)
    )).toDF("sentence", "label")

    //在spark ml中对应的转换器帮助我们进行转换
    /**
     * +------------------------------+-----+-------------------------------------+
     * |sentence                      |label|words                                |
     * +------------------------------+-----+-------------------------------------+
     * |Hadoop is a big data framework|1    |[hadoop, is, a, big, data, framework]|
     * |Spark is a big data framework |0    |[spark, is, a, big, data, framework] |
     * |Spark is a big data framework |0    |[spark, is, a, big, data, framework] |
     * +------------------------------+-----+-------------------------------------+
     */
    val tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words")
    val wordsDF = tokenizer.transform(df)

    //需要对wordsDF进行转换，就是进行TF词频统计
    /**
     * +------------------------------+-----+-------------------------------------+----------------------------------------------------------------------------+
     * |sentence                      |label|words                                |tf                                                                          |
     * +------------------------------+-----+-------------------------------------+----------------------------------------------------------------------------+
     * |Hadoop is a big data framework|1    |[hadoop, is, a, big, data, framework]|(262144,[15889,30006,155117,160735,227410,234887],[1.0,1.0,1.0,1.0,1.0,1.0])|
     * |Spark is a big data framework |0    |[spark, is, a, big, data, framework] |(262144,[15889,30006,160735,227410,234657,234887],[1.0,1.0,1.0,1.0,1.0,1.0])|
     * |Spark is a big data framework |0    |[spark, is, a, big, data, framework] |(262144,[15889,30006,160735,227410,234657,234887],[1.0,1.0,1.0,1.0,1.0,1.0])|
     * +------------------------------+-----+-------------------------------------+----------------------------------------------------------------------------+
     */
    val hashingTF = new HashingTF().setInputCol(tokenizer.getOutputCol).setOutputCol("tf")
    //设置特征数目，一个单词代表一个特征，默认1<<18 也就是262144
    //.setNumFeatures(262144)
    val tfDF = hashingTF.transform(wordsDF)

    //为了过滤那些停用词，那么需要进行TF*IDF，
    //这个IDF不是转换器，而是模型学习器（算法）
    val idf = new IDF().setInputCol(hashingTF.getOutputCol).setOutputCol("features")
    //注意学习器调用fit方法后，拿到的是转换器
    val idfModel = idf.fit(tfDF)
    idfModel.transform(tfDF).show(false)

    spark.stop()
  }
}
