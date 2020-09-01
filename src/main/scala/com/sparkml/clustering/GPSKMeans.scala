package com.sparkml.clustering

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.{DoubleType, StringType, StructField, StructType}

/**
 * 聚类算法--K-means聚类算法--基于用户位置信息的商业选址案例
 */
object GPSKMeans {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.WARN)
    val spark = SparkSession.builder().appName("").master("local").getOrCreate()
    //加载数据
    val schema = StructType(List(
      //1,30.624806,104.136604,211846
      StructField("userId", StringType, true),
      StructField("lat", DoubleType, true), //纬度
      StructField("lon", DoubleType, true), //经度
      StructField("timestamp", StringType, true)
    ))
    val df = spark.read.schema(schema).csv("data/clustering/gps.csv")
    val featuresRDD = df.select("lat", "lon").rdd.map(row => {
      val lat = row.getAs[Double]("lat")
      val lon = row.getAs[Double]("lon")
      Vectors.dense(lat, lon)
    })

    featuresRDD.cache()
    //将样本划分成训练集和测试
    val Array(trainRDD, testRDD) = featuresRDD.randomSplit(Array(0.7, 0.3))

    /**
     * def train(
     * data: RDD[Vector],//训练集数据，注意聚类算法，是无监督学习方法，没有类别数据，只有特征
     * k: Int,//需要将记录根据特征划分成三个类别，也就是有三个类中心
     * maxIterations: Int//在求解三个类中心最优位置时，需要迭代的次数
     * ): KMeansModel
     */
    //这个模型包含三个类中心点，三个类中心点都带有了角标，角标从（0，k-1）
    val kMeansModel = KMeans.train(trainRDD, 3, 20)
    //打印k个类中心点
    println(kMeansModel.clusterCenters.mkString("==="))
    //可以进行预测,返回的是类中心对应的角标
    //kMeansModel.predict(testRDD)
    val kIndex = kMeansModel.predict(Vectors.dense(30.624806, 104.136604))
    println(kIndex + "===" + kMeansModel.clusterCenters(kIndex))

    spark.stop()
  }
}
