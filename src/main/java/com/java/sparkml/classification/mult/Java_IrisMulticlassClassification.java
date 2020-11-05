package com.java.sparkml.classification.mult;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS;
import org.apache.spark.mllib.classification.NaiveBayes;
import org.apache.spark.mllib.classification.NaiveBayesModel;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.DecisionTree;
import org.apache.spark.mllib.tree.RandomForest;
import org.apache.spark.mllib.tree.model.DecisionTreeModel;
import org.apache.spark.mllib.tree.model.RandomForestModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.util.Utils;
import scala.Tuple2;

import java.util.HashMap;
import java.util.Map;

/**
 * @Author:
 * @Date: 2020-11-03 17:14
 * @Version: 1.0
 * @Modified By:
 * @Description:
 */
/**
 * 分类算法--多分类算法--鸢尾花案例
 */
public class Java_IrisMulticlassClassification {
    public static void main(String[] args) {
        // todo 1.加载数据
        Logger.getLogger("org").setLevel(Level.WARN);
        SparkSession spark = SparkSession.builder().appName("IrisMulticlassClassification").master("local").getOrCreate();
        //加载鸢尾花数据
        Dataset<Row> df = spark.read().option("inferSchema", "true").csv("data/classification/iris.data");
        //df.printSchema()
        df.show(10);  // 测试加载数据是否正常

        // todo 2.提取特征工程
        //LabeledPoint（类别，特征向量）
        //提取出鸢尾花的类别，然后给每个类别加上index角标，以index角标作为label
        Map<String, Long> labelMap = df.javaRDD().map((Row row) -> row.getString(4)).distinct().zipWithIndex().collectAsMap();
        //println(labelMap)

        JavaSparkContext jsc = new JavaSparkContext(spark.sparkContext());
        Broadcast<Map<String, Long>> labelMapBroadCast = jsc.broadcast(labelMap);

        //Map(Iris-virginica -> 2, Iris-setosa -> 0, Iris-versicolor -> 1)
        //提取鸢尾花特征
        JavaRDD<LabeledPoint> labeledPointRDD = df.javaRDD().map((Row row) -> {
            //5.8,4.0,1.2,0.2,Iris-setosa
            double heX = row.getDouble(0);
            double heY = row.getDouble(1);
            double hbX = row.getDouble(2);
            double hbY = row.getDouble(3);
            double label = (double) labelMapBroadCast.value().get(row.get(4));
            return new LabeledPoint(label, Vectors.dense(new double[]{heX, heY, hbX, hbY}));
        });

        labeledPointRDD.cache();

        //将数据划分成训练集和测试集
        JavaRDD<LabeledPoint>[] needFitData = labeledPointRDD.randomSplit(new double[]{0.8, 0.2});
        JavaRDD<LabeledPoint> trainRDD = needFitData[0];
        JavaRDD<LabeledPoint> testRDD = needFitData[1];
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
        LogisticRegressionModel lrModel = new LogisticRegressionWithLBFGS().setNumClasses(labelMap.size()).run(trainRDD.rdd());
        JavaRDD<Tuple2<Object, Object>> lrPredictAndActualRDD = testRDD.map((LabeledPoint labeledPoint) -> {
                double predict = lrModel.predict(labeledPoint.features());
                Tuple2<Object, Object> tp = new Tuple2<>(predict, labeledPoint.label());
                return tp;
        });

        //需要多分类评估器，对模型进行评估
        MulticlassMetrics lrMetrics = new MulticlassMetrics(lrPredictAndActualRDD.rdd());
        //对多分类模型进行评估，可以使用准确率，精准率，召回率
        //准确率 预测正取的总数/需要预测的总数
        System.out.println("逻辑回归 准确率accuracy = " + lrMetrics.accuracy() + "\t精确率weightedPrecision:"
                + lrMetrics.weightedPrecision() + "\t召回率weightedRecall:" + lrMetrics.weightedRecall());

        //todo 使用决策树进行预测
        DecisionTreeModel dtModel = DecisionTree.trainClassifier(
                trainRDD,
                labelMap.size(),
                new HashMap<>(),
                "gini", 8, 2);
        JavaRDD<Tuple2<Object, Object>> dtPredictAndActualRDD = testRDD.map((LabeledPoint lp) -> {
            double predict = dtModel.predict(lp.features());
            return new Tuple2<>(predict, lp.label());
        });
        //需要多分类评估器，对模型进行评估
        MulticlassMetrics dtMetrics = new MulticlassMetrics(dtPredictAndActualRDD.rdd());
        //对多分类模型进行评估，可以使用准确率，精准率，召回率
        //准确率 预测正取的总数/需要预测的总数
        System.out.println("决策树 准确率accuracy = " + dtMetrics.accuracy() + "\t精确率weightedPrecision:"
                + dtMetrics.weightedPrecision() + "\t召回率weightedRecall:" + dtMetrics.weightedRecall());

        //todo 使用随机森林进行预测
        RandomForestModel rfModel = RandomForest.trainClassifier(
                trainRDD,
                labelMap.size(),
                new HashMap<>(), 5, "auto", "gini", 5, 2, Utils.random().nextInt());
        //使用模型进行预测
        JavaRDD<Tuple2<Object, Object>> rfPredictAndActualRDD = testRDD.map((LabeledPoint lp) -> {
                double predict = rfModel.predict(lp.features());
                return new Tuple2<>(predict, lp.label());
        });
        //需要多分类评估器，对模型进行评估
        MulticlassMetrics rfMetrics = new MulticlassMetrics(rfPredictAndActualRDD.rdd());
        //对多分类模型进行评估，可以使用准确率，精准率，召回率
        //准确率 预测正取的总数/需要预测的总数
        System.out.println("随机森林 准确率accuracy = " + rfMetrics.accuracy() + "\t精确率weightedPrecision:"
                + rfMetrics.weightedPrecision() + "\t召回率weightedRecall:" + rfMetrics.weightedRecall());

        //todo 贝叶斯算法进行预测
        NaiveBayesModel nbModel = NaiveBayes.train(trainRDD.rdd());
        //使用模型进行预测
        JavaRDD<Tuple2<Object, Object>> nbPredictAndActualRDD = testRDD.map(new Function<LabeledPoint, Tuple2<Object, Object>>() {
            @Override
            public Tuple2<Object, Object> call(LabeledPoint lp) throws Exception {
                double predict = nbModel.predict(lp.features());
                return new Tuple2<>(predict, lp.label());
            }
        });
        //需要多分类评估器，对模型进行评估
        MulticlassMetrics nbMetrics = new MulticlassMetrics(nbPredictAndActualRDD.rdd());
        //对多分类模型进行评估，可以使用准确率，精准率，召回率
        //准确率 预测正取的总数/需要预测的总数
        System.out.println("贝叶斯 准确率accuracy = " + nbMetrics.accuracy() + "\t精确率weightedPrecision:"
                + nbMetrics.weightedPrecision() + "\t召回率weightedRecall:" + nbMetrics.weightedRecall());

        spark.stop();
    }
}
