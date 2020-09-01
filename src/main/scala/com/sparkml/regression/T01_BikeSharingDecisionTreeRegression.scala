package com.sparkml.regression

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.configuration.Algo.Algo
import org.apache.spark.mllib.tree.configuration.{Algo, BoostingStrategy, Strategy}
import org.apache.spark.mllib.tree.impurity.Variance
import org.apache.spark.mllib.tree.loss.SquaredError
import org.apache.spark.mllib.tree.{DecisionTree, GradientBoostedTrees, RandomForest}
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.sql.SparkSession

/**
 * 回归算法--决策树回归--共享单车案例--
 * 使用决策树算法DecisionTree、随机森林算法RandomForest、梯度提升树算法GradientBoostedTrees
 * 回归模型预测每小时单车租用数量
 */
object T01_BikeSharingDecisionTreeRegression {
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
    val df2 = df.select(
      //这八个字段是类别特征，需要进行额外处理
      "season", "yr", "mnth", "hr", "holiday", "weekday", "workingday", "weathersit",
      //这四个特征已经被正则化处理过，直接使用即可
      "temp", "atemp", "hum", "windspeed",
      //标签
      "cnt"
    )
    //无论是分类还是回归，训练模型的数据都是LabeledPoint(label,features)
    val labeledPointRDD = df2.rdd.map(row => {
      //八个类别特征
      val categoryFeatures = Array(row.getString(0).toInt - 1, row.getString(1), row.getString(2).toInt - 1, row.getString(3), row.getString(4), row.getString(5), row.getString(6), row.getString(7).toInt - 1).map(_.toString.toDouble)
      //其他四个特征
      val otherFeatures = Array(row.getString(8), row.getString(9), row.getString(10), row.getString(11)).map(_.toDouble)
      //标签
      val label = row.getString(12).toDouble
      LabeledPoint(label, Vectors.dense(categoryFeatures ++ otherFeatures))
    })

    labeledPointRDD.cache()

    //labeledPointRDD.foreach(println)
    //将数据划分成训练集和测试集
    val Array(trainRDD, testRDD) = labeledPointRDD.randomSplit(Array(0.7, 0.3))

    /**
     * 决策树回归
     * 2.1 决策树回归
     * 2.2 随机森林random forests
     * 2.2 梯度提升树gradient-boosted trees
     */
    //todo 使用决策树算法对数据进行训练得到模型，决策树是可以支持类别特征进行训练，可以不用处理

    /**
     * def trainRegressor(
     * input: RDD[LabeledPoint],//训练数据
     * categoricalFeaturesInfo: Map[Int, Int],//训练数据中，其特征值是否包含类别特征，如果有，那么需要告知算法
     * impurity: String,//不纯度度量方式，是决策树选取特征的一个重要参考依据，在回归问题中，只支持variance方差度量方式
     * maxDepth: Int,//数的深度
     * maxBins: Int//最大叶子数，最大叶子数一定要大于类别特征中的数量
     * ): DecisionTreeModel
     */
    /**
     * * season		季节[1,2,3,4]
     * * yr			年份[0,1]    0代表2011年  1代表2012年
     * * mnth		月份[1,2,3,4,5,6,7,8,9,10,11,12]
     * * hr			小时[0,1,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,2,20,21,22,23]
     * * holiday		是否是节假日[0,1]
     * * weekday		一周第几天[0,1,2,3,4,5,6]
     * * workingday	是否是工作日[0,1]
     * * weathersit  天气状况[1,2,3,4]  Array(1,0,0,0),Array(0,1,0,0,0)
     */
    val dtModel: DecisionTreeModel = DecisionTree.trainRegressor(trainRDD, Map[Int, Int](0 -> 4, 1 -> 2, 2 -> 12, 3 -> 24, 4 -> 2, 5 -> 7, 6 -> 2, 7 -> 4),
      "variance", 6, 32)
    //进行预测,返回元组（预测值，实际值）
    val predictAndActualRDD = testRDD.map { case LabeledPoint(label, features) => (dtModel.predict(features), label) }
    predictAndActualRDD.foreach(println)
    //需要有一个回归模型评估器，对模型预测的结果进行评估,回归模型的参考指标RMSE(均方根误差)，MAE (Mean Absolute Error 平均绝对值误差)
    val dtMetrics = new RegressionMetrics(predictAndActualRDD)
    println("决策树 RMSE=" + dtMetrics.rootMeanSquaredError)
    println("决策树 MAE=" + dtMetrics.meanAbsoluteError)


    //todo 使用随机森林random forests算法对每小时共享单车租用数量进行预测
    /**
     * def trainRegressor(
     * input: RDD[LabeledPoint],//训练集数据
     * categoricalFeaturesInfo: Map[Int, Int],//样本特征中是否包含特征数据，包含需要告知
     * numTrees: Int,//随机森林中有几颗决策树
     * featureSubsetStrategy: String,//随机森林中决策树在进行训练模型的时候选取的特征数目   "auto", "all", "sqrt", "log2",
     * impurity: String,//不纯度度量方式，仅支持方差度量
     * maxDepth: Int,//数的最大深度
     * maxBins: Int,//数据的最大叶子数，如果样本中包含了类别特征，那么叶子数一定要大于类别特征中最大的那个值
     * seed: Int = Utils.random.nextInt()): RandomForestModel
     */
    val rfModel = RandomForest.trainRegressor(trainRDD, Map[Int, Int](0 -> 4, 1 -> 2, 2 -> 12, 3 -> 24, 4 -> 2, 5 -> 7, 6 -> 2, 7 -> 4),
      8, "onethird", "variance", 6, 32
    )
    //进行预测,返回元组（预测值，实际值）
    val rfpredictAndActualRDD = testRDD.map { case LabeledPoint(label, features) => (rfModel.predict(features), label) }
    //需要有一个回归模型评估器，对模型预测的结果进行评估,回归模型的参考指标RMSE(均方根误差)，MAE (Mean Absolute Error 平均绝对值误差)
    val rfMetrics = new RegressionMetrics(rfpredictAndActualRDD)
    println("随机森林 RMSE=" + rfMetrics.rootMeanSquaredError)
    println("随机森林 MAE=" + rfMetrics.meanAbsoluteError)

    //todo 使用梯度提升树gradient-boosted trees算法进行预测
    /**
     * def train(
     * input: RDD[LabeledPoint],//训练集数据
     * boostingStrategy: BoostingStrategy//梯度提升树提升策略
     * ): GradientBoostedTreesModel
     */
    val strategy = new Strategy(Algo.Regression, Variance, 6)
    strategy.maxBins = 32
    strategy.categoricalFeaturesInfo = Map[Int, Int](0 -> 4, 1 -> 2, 2 -> 12, 3 -> 24, 4 -> 2, 5 -> 7, 6 -> 2, 7 -> 4)
    val gbtMode = GradientBoostedTrees.train(trainRDD, BoostingStrategy(strategy, SquaredError))
    //进行预测,返回元组（预测值，实际值）
    val gbtpredictAndActualRDD = testRDD.map { case LabeledPoint(label, features) => (gbtMode.predict(features), label) }
    //需要有一个回归模型评估器，对模型预测的结果进行评估,回归模型的参考指标RMSE(均方根误差)，MAE (Mean Absolute Error 平均绝对值误差)
    val gbtMetrics = new RegressionMetrics(gbtpredictAndActualRDD)
    println("梯度提升树 RMSE=" + gbtMetrics.rootMeanSquaredError)
    println("梯度提升树 MAE=" + gbtMetrics.meanAbsoluteError)

    spark.stop()


  }
}
