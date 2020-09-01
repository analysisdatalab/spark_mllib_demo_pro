package com.sparkml.regression

import org.apache.log4j.{Level, Logger}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.{LabeledPoint, LassoWithSGD, RidgeRegressionWithSGD}
import org.apache.spark.sql.SparkSession

import scala.collection.mutable.ArrayBuffer

/**
 * 回归算法--线性回归--共享单车案例--使用RidgeRegressionWithSGD算法回归模型预测每小时单车租用数量
 */
object T04_BikeSharingRidgeRegression {


  /**
   * 将类别特征转换成向量
   *
   * @param categoryMap Map(1->0,2->1,3->2,4->3)
   * @param feature     1,2,3,4
   */
  def transformatFeatures(categoryMap: collection.Map[String, Long], feature: String) = {
    val featuresArray = new Array[Double](categoryMap.size)
    val index = categoryMap(feature).toInt
    featuresArray(index) = 1.0
    featuresArray
  }

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.WARN)
    val spark = SparkSession.builder().appName("BikeSharingRegression").master("local").getOrCreate()

    //todo 加载数据
    val df = spark.read.option("header", "true").csv("data/regression/bikesharing/hour.csv")

    /**
     * instant		序号
     * dteday		年月日时分秒日期格式
     * season		季节[1,2,3,4]
     * yr			年份[0,1]    0代表2011年  1代表2012年
     * mnth		月份[1,2,3,4,5,6,7,8,9,10,11,12]
     * hr			小时[0,1,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,2,20,21,22,23]
     * holiday		是否是节假日[0,1]
     * weekday		一周第几天[0,1,2,3,4,5,6]
     * workingday	是否是工作日[0,1]
     * weathersit  天气状况[1,2,3,4]  Array(1,0,0,0),Array(0,1,0,0,0)
     * temp		气温
     * atemp		体感温度
     * hum			湿度
     * windspeed	方向
     * casual		没有注册的用户租用自行车的数量
     * registered	注册的用户租用自行的数量
     * cnt			总的租用自行车的数量
     */
    //todo 提取共享单车租用信息特征
    val recordRDD = df.select(
      //这八个字段是类别特征，需要进行额外处理
      "season", "yr", "mnth", "hr", "holiday", "weekday", "workingday", "weathersit",
      //这四个特征已经被正则化处理过，直接使用即可
      "temp", "atemp", "hum", "windspeed",
      //标签
      "cnt"
    ).rdd

    val bigMap = (0 to 7).map(index => {
      val smallMap = recordRDD.map(row => row.getString(index)).distinct().zipWithIndex().collectAsMap()
      //（特征对应的角标， 是一个map封装了特征类别数据以及对应的角标）
      (index.toString, smallMap)
    }).toMap
    val bigMapBroadCast: Broadcast[Map[String, collection.Map[String, Long]]] = spark.sparkContext.broadcast(bigMap)

    //无论是分类还是回归，训练模型的数据都是LabeledPoint(label,features)
    val labeledPointRDD = recordRDD.map(row => {
      //每一遍历到一条数据就是一个row，row中包含8个类别特征数据，我们要将类别特征的数据转换成向量（数组）
      val featureBuffer = ArrayBuffer[Double]()
      (0 to 7).foreach(index => {
        //获取每个字段的值
        val fieldValue = row.getString(index)
        //获取广播变量中对应字段的smallMap
        val bigMap2 = bigMapBroadCast.value
        //从最外层的Map中，通过字段索引将里面的smallMap取出来
        val smallMap = bigMap2(index.toString)
        //将类别特征的值转换成向量
        featureBuffer ++= transformatFeatures(smallMap, fieldValue)
      })
      //其他四个特征
      val otherFeatures = Array(row.getString(8), row.getString(9), row.getString(10), row.getString(11)).map(_.toDouble)
      featureBuffer ++= otherFeatures
      //标签
      val label = row.getString(12).toDouble
      LabeledPoint(label, Vectors.dense(featureBuffer.toArray))
    })

    labeledPointRDD.cache()

    //labeledPointRDD.foreach(println)
    //将数据划分成训练集和测试集
    val Array(trainRDD, testRDD) = labeledPointRDD.randomSplit(Array(0.7, 0.3))

    //todo 使用线性回归算法，对共享单车每小时出租数量进行预测
    /**
     * def train(
     * input: RDD[LabeledPoint],//训练集数据
     * numIterations: Int//迭代次数 默认迭代次数是100次
     * ): LinearRegressionModel
     */
    //val lrModel = LinearRegressionWithSGD.train(trainRDD, 100)
    /**
     * def train(
     * input: RDD[LabeledPoint],//训练集数据
     * numIterations: Int,//迭代次数,随机梯度训练次数,默认100
     * stepSize: Double,//与上一次训练的步长，不能过大
     * regParam: Double,//正则化项参数
     * miniBatchFraction: Double//每次继续梯度下降训练时所使用的的数据集
     * ): RidgeRegressionModel
     */
 val ridgeModle = RidgeRegressionWithSGD.train(trainRDD, 100, 1.0, 0.01, 1.0)

    //进行预测,返回元组（预测值，实际值）
    val ridgePredictAndActualRDD = testRDD.map { case LabeledPoint(label, features) => (ridgeModle.predict(features), label) }
    //需要有一个回归模型评估器，对模型预测的结果进行评估,回归模型的参考指标RMSE(均方根误差)，MAE (Mean Absolute Error 平均绝对值误差)
    val ridgeMetrics = new RegressionMetrics(ridgePredictAndActualRDD)
    println("ridge回归 RMSE=" + ridgeMetrics.rootMeanSquaredError)
    println("ridge回归 MAE=" + ridgeMetrics.meanAbsoluteError)

    spark.stop()


  }
}
